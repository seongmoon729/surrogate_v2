import ray
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.data import transforms as T
from detectron2.structures import ImageList
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.events import EventStorage
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

import compressai.zoo as ca_zoo
from compressai.layers import GDN
from compressai.models.utils import deconv

import codec_ops


class EndToEndNetwork(nn.Module):
    def __init__(self, surrogate_quality, vision_task, od_cfg=None, input_format='BGR'):
        super().__init__()
        assert input_format in ['RGB', 'BGR']
        self.surrogate_quality = surrogate_quality
        self.vision_task = vision_task
        self.od_cfg = od_cfg
        self.input_format = input_format

        self.surrogate_network = ca_zoo.mbt2018(self.surrogate_quality, pretrained=True)
        self.filtering_network = FilteringNetwork(self.surrogate_network)
        self.vision_network = VisionNetwork(self.vision_task, self.od_cfg)

        self.inference_aug = T.ResizeShortestEdge(
            [od_cfg.INPUT.MIN_SIZE_TEST, od_cfg.INPUT.MIN_SIZE_TEST], od_cfg.INPUT.MAX_SIZE_TEST
        )

    def forward(self, inputs, eval_codec=None, eval_quality=None, eval_downscale=None, eval_filtering=True):
        """ Forward method. 
            Pre-fixed arguments with 'eval' are only for inference mode (.eval())
        """
        if self.vision_task == 'classification':
            pass
        else:
            if not self.training:
                if eval_codec: assert eval_quality
                return self.inference(inputs, eval_codec, eval_quality, eval_downscale, eval_filtering)

            # Preprocess inputs.
            images = self.preprocess_image_for_od(inputs)

            # Normalize & filter image.
            images.tensor, (h, w) = self.filtering_network.preprocess(images.tensor)
            images.tensor = self.filtering_network(images.tensor / 255.)

            # Process image with codec.
            codec_out = self.surrogate_network(images.tensor)
            images.tensor = self.filtering_network.postprocess(codec_out['x_hat'], (h, w))
            bpp = self.compute_bpp(codec_out)
            loss_r = torch.mean(bpp)

            # Convert RGB to BGR & Denormalize.
            images.tensor = images.tensor[:, [2, 1, 0], :, :] * 255.

            # Detect objects.
            if 'instances' in inputs[0]:
                gt_instances = [x['instances'].to(self.device) for x in inputs]
            else:
                gt_instances = None            
            if 'proposals' in inputs[0]:
                proposals = [x['proposals'].to(self.device) for x in inputs]
            else:
                proposals = None

            losses_d = self.vision_network(images, gt_instances, proposals)
            loss_d = sum(losses_d.values())

            losses = dict()
            losses['r'] = loss_r
            losses['d'] = loss_d
            return losses

    def inference(self, original_image, codec, quality, downscale, filtering=True):
        """ This method processes only one image (not batched!). """
        assert not self.training
        assert isinstance(original_image, np.ndarray)
        assert len(original_image.shape) == 3

        results = dict()

        with torch.no_grad():
            if self.vision_task == 'classification':
                pass
            else:
                original_image = original_image.astype('float32')
                if self.input_format == 'BGR':  # BGR -> RGB.
                    original_image = original_image[:, :, ::-1]

                height, width = original_image.shape[:2]
                image = self.inference_aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype('float32').transpose(2, 0, 1), device=self.device)

                # Filter.
                image = image[None, ...]
                image, (h, w) = self.filtering_network.preprocess(image)
                if filtering:
                    filtered_image = self.filtering_network(image / 255.)
                else:
                    filtered_image = image / 255.

                # Encode & Decode
                if codec:
                    filtered_image = self.filtering_network.postprocess(filtered_image, (h, w))
                    filtered_image = filtered_image[0].detach().cpu()
                    reconstructed_image, bpp = ray.get(codec_ops.ray_codec_fn.remote(
                        filtered_image.numpy(),
                        codec=codec,
                        quality=quality,
                        downscale=downscale))
                    reconstructed_image = torch.as_tensor(reconstructed_image, device=self.device)
                else:
                    codec_out = self.surrogate_network(filtered_image)
                    filtered_image = self.filtering_network.postprocess(filtered_image[0].detach().cpu(), (h, w))
                    reconstructed_image = self.filtering_network.postprocess(codec_out['x_hat'][0], (h, w))
                    bpp = self.compute_bpp(codec_out).item()

                # Detector takes 'BGR' format image of range [0, 255].
                vision_inputs = {'image': reconstructed_image[[2, 1, 0], :, :] * 255., 'height': height, 'width': width}
                vision_results = self.vision_network([vision_inputs])[0]
        
        results.update(vision_results)
        results.update({'bpp': bpp})
        results.update({
            'image': {
                'filtered': filtered_image.cpu(),
                'reconstructed': reconstructed_image.cpu(),
            }
        })
        return results

    def preprocess_image_for_od(self, batched_inputs):
        images = [x['image'] for x in batched_inputs]

        # BGR -> RGB
        if self.input_format == 'BGR':
            images = [x[[2, 1, 0], :, :] for x in images]

        images = [x.to(self.device) for x in images]
        images = ImageList.from_tensors(images, self.vision_network.size_divisibility)
        return images

    def compute_bpp(self, out):
        size = out['x_hat'].size()
        num_pixels = size[-2] * size[-1]
        return sum(-torch.log2(likelihoods).sum(axis=(1, 2, 3)) / num_pixels
                for likelihoods in out['likelihoods'].values())

    @property
    def device(self):
        return self.vision_network.device


class FilteringNetwork(nn.Module):
    def __init__(self, surrogate_network):
        super().__init__()
        self.surrogate_network = surrogate_network
        self.surrogate_encoder = SurrogateEncoder(self.surrogate_network)

        # TODO: Is this module necessary??
        M, N = self.surrogate_network.M, self.surrogate_network.N
        self.pixel_rate_estimator = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N // 2, kernel_size=5, stride=2),
            GDN(N // 2, inverse=True),
            deconv(N // 2, N // 4, kernel_size=5, stride=2),
            GDN(N // 4, inverse=True),
            deconv(N // 4, 16, kernel_size=5, stride=2),
        )

        self.filter = nn.Sequential(
            FilteringBlock(19, 64),
            FilteringBlock(64, 64),
            FilteringBlock(64, 64),
            FilteringBlock(64, 64),
            # nn.Conv2d(64, 3, kernel_size=3, stride=1, padding='same'),
            nn.Conv2d(64, 3, kernel_size=1, stride=1),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.surrogate_encoder(x)
        out = self.pixel_rate_estimator(out)
        out = torch.cat([x, out], axis=1)
        out = self.filter(out)
        out = out + x
        out = torch.clip(out, 0., 1.)
        return out

    def preprocess(self, x):
        # Pad.
        def _check_for_padding(x):
            remainder = x % 64
            if remainder:
                return 64 - remainder
            return remainder
        h, w = x.shape[-2:]
        h_pad, w_pad = map(_check_for_padding, (h, w))
        x = F.pad(x, [0, w_pad, 0, h_pad])
        return x, (h, w)
        
    def postprocess(self, x, size):
        # Unpad.
        h, w = size
        x = x[..., :h, :w]

        # Clip [0, 1]
        x = x.clip(0., 1.)
        return x


class VisionNetwork(nn.Module):
    def __init__(self, task, od_cfg):
        super().__init__()
        self.task = task
        self.od_cfg = od_cfg
        self._event_storage = EventStorage(0)

        if self.task == 'classification':
            pass
        else:
            assert od_cfg
            self.model = build_object_detection_model(od_cfg)
    
    def forward(self, images, gt_instances=None, proposals=None):
        if self.task == 'classification':
            pass
        else:
            if not self.training:
                return self.inference(images)
            
            with self._event_storage:
                images.tensor = self.preprocess(images.tensor)
                features = self.model.backbone(images.tensor)

                if proposals is None:
                    proposals, proposal_losses = self.model.proposal_generator(images, features, gt_instances)
                else:
                    proposal_losses = {}
                
                _, detector_losses = self.model.roi_heads(images, features, proposals, gt_instances)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

    def inference(self, batched_inputs):
        assert not self.training

        if self.task == 'classification':
            pass
        else:
            images = [x['image'].to(self.device) for x in batched_inputs]
            images = [self.preprocess(x) for x in images]
            images = ImageList.from_tensors(images, self.model.backbone.size_divisibility)

            features = self.model.backbone(images.tensor)
            proposals, _ = self.model.proposal_generator(images, features, None)
            results, _  = self.model.roi_heads(images, features, proposals, None)
            results = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
            return results

    def preprocess(self, x):
        if self.task == 'classification':
            pass
        else:
            x = (x - self.model.pixel_mean) / self.model.pixel_std
            return x

    @property
    def device(self):
        return self.model.device

    @property
    def size_divisibility(self):
        return self.model.backbone.size_divisibility


class SurrogateEncoder(nn.Module):
    def __init__(self, surrogate_network):
        super().__init__()
        self.surrogate_network = surrogate_network

    def forward(self, x):
        y = self.surrogate_network.g_a(x)
        z = self.surrogate_network.h_a(y)
        z_hat, _ = self.surrogate_network.entropy_bottleneck(z)
        params = self.surrogate_network.h_s(z_hat)

        y_hat = self.surrogate_network.gaussian_conditional.quantize(y, "dequantize")
        ctx_params = self.surrogate_network.context_prediction(y_hat)
        gaussian_params = self.surrogate_network.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.surrogate_network.gaussian_conditional(y, scales_hat, means=means_hat)
        return y_likelihoods
    

class FilteringBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels),
        )
        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.proj:
            x = self.proj(x)
        out = out + x
        out = self.relu(out)
        return out


def build_object_detection_model(cfg):
    cfg = cfg.clone()
    model = GeneralizedRCNN(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    return model
