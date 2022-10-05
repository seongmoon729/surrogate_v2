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

import codec_ops


class EndToEndNetwork(nn.Module):
    def __init__(self, surrogate_quality, vision_task, norm_layer='cn', od_cfg=None, input_format='BGR'):
        super().__init__()
        assert input_format in ['RGB', 'BGR']
        self.surrogate_quality = surrogate_quality
        self.vision_task = vision_task
        self.norm_layer = norm_layer
        self.od_cfg = od_cfg
        self.input_format = input_format

        # Networks.
        self.surrogate_network = ca_zoo.mbt2018(self.surrogate_quality, pretrained=True)
        self.filtering_network = FilteringNetwork(self.norm_layer)
        self.vision_network = VisionNetwork(self.vision_task, self.od_cfg)

        self.inference_aug = T.ResizeShortestEdge(
            [od_cfg.INPUT.MIN_SIZE_TEST, od_cfg.INPUT.MIN_SIZE_TEST], od_cfg.INPUT.MAX_SIZE_TEST
        )

    def forward(self, inputs, eval_codec=None, eval_quality=None, eval_downscale=None, eval_filtering=False):
        """ Forward method. 
            Pre-fixed arguments with 'eval' are only for inference mode (.eval())
        """
        if self.vision_task == 'classification':
            pass
        else:
            if not self.training:
                return self.inference(inputs, eval_codec, eval_quality, eval_downscale, eval_filtering)

            # Convert input format to RGB & batch the images after applying padding.
            images = self.preprocess_image_for_od(inputs)

            # Normalize & filter.
            images.tensor, (h, w) = self.filtering_network.preprocess(images.tensor)
            images.tensor = self.filtering_network(images.tensor / 255.)

            # Apply codec.
            codec_out = self.surrogate_network(images.tensor)
            images.tensor = self.filtering_network.postprocess(codec_out['x_hat'], (h, w))

            # Compute averaged bit rate & use it as rate loss.
            loss_r = torch.mean(self.compute_bpp(codec_out))

            # Convert RGB to BGR & denormalize.
            images.tensor = images.tensor[:, [2, 1, 0], :, :] * 255.

            if 'instances' in inputs[0]:
                gt_instances = [x['instances'].to(self.device) for x in inputs]
            else:
                gt_instances = None            
            if 'proposals' in inputs[0]:
                proposals = [x['proposals'].to(self.device) for x in inputs]
            else:
                proposals = None

            # Internally normalize input & compute losses for object detection.
            losses_d = self.vision_network(images, gt_instances, proposals)
            loss_d = sum(losses_d.values())

            losses = dict()
            losses['r'] = loss_r
            losses['d'] = loss_d
            return losses

    def inference(self, original_image, codec, quality, downscale, filtering):
        """ This method processes only one image (not batched!). """
        assert not self.training
        assert isinstance(original_image, np.ndarray)
        assert len(original_image.shape) == 3

        results = dict()

        with torch.no_grad():
            if self.vision_task == 'classification':
                pass
            else:
                # If image format is 'BGR', convert to 'RGB'.
                if self.input_format == 'BGR':
                    original_image = original_image[:, :, ::-1]
                
                # Convert dtype to 'float32' & normalize to [0, 1].
                original_image = original_image.astype('float32') / 255.

                # Convert (H, W, C) format to (C, H, W),
                # which is canonical input format of torch models.
                original_image = original_image.transpose(2, 0, 1)

                # Convert to torch tensor.
                original_image = torch.as_tensor(original_image, device=self.device)

                # 1. Apply filtering or not.
                if filtering:
                    padded_image, (h, w) = self.filtering_network.preprocess(original_image)
                    filtered_image = self.filtering_network(padded_image[None, ...])[0]
                    filtered_image = self.filtering_network.postprocess(filtered_image, (h, w))
                else:
                    filtered_image = original_image

                # Convert torch tensor to numpy array.
                filtered_image = filtered_image.detach().cpu().numpy()

                # 2. Apply codec.
                if codec == 'none':
                    # (a). without codec.
                    reconstructed_image, bpp = filtered_image, 24.
                elif codec == 'surrogate':
                    # (b). surrogate codec.
                    filtered_image_ = torch.as_tensor(filtered_image, device=self.device)
                    filtered_image_, (h, w) = self.filtering_network.preprocess(filtered_image_)
                    codec_out = self.surrogate_network(filtered_image_[None, ...])
                    # Unpad & cal
                    reconstructed_image, bpp = (
                        self.filtering_network.postprocess(codec_out['x_hat'][0], (h, w)),
                        self.compute_bpp(codec_out).item())
                    reconstructed_image = reconstructed_image.detach().cpu().numpy()
                else:
                    # (c). conventional codec.
                    reconstructed_image, bpp = ray.get(codec_ops.ray_codec_fn.remote(
                        filtered_image,
                        codec=codec,
                        quality=quality,
                        downscale=downscale))

                # Convert reconstructed image format to (H, W, C) & denormalize.
                od_input_image = reconstructed_image.transpose(1, 2, 0) * 255.

                # Convert RGB to BGR.
                od_input_image = od_input_image[:, :, ::-1]

                # Convert dtype to 'uint8'
                od_input_image = od_input_image.round().astype('uint8')

                # Store original size.
                height, width = od_input_image.shape[:2]

                # Augment reconstructed image for detection network.
                od_input_image = (self.inference_aug.get_transform(od_input_image)
                                      .apply_image(od_input_image))

                # Convert numpy array to torch tensor & change format to (C, H, W)
                od_input_image = torch.as_tensor(
                    od_input_image.astype('float32').transpose(2, 0, 1), device=self.device)

                # Detector takes 'BGR' format image of range [0, 255].
                vision_inputs = {'image': od_input_image, 'height': height, 'width': width}
                vision_results = self.vision_network([vision_inputs])[0]
        
        results.update(vision_results)
        results.update({'bpp': bpp})
        results.update({
            'image': {
                # Change returned numpy array format to (H, W, C).
                'filtered': filtered_image.transpose(1, 2, 0),
                'reconstructed': reconstructed_image.transpose(1, 2, 0),
            }
        })
        return results

    def preprocess_image_for_od(self, batched_inputs):
        """ Batch the images after padding. """
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

    def train(self, mode=True):
        super().train(mode)
        self.model.backbone.train(False)


class FilteringNetwork(nn.Module):
    def __init__(self, norm_layer='cn'):
        super().__init__()
        self.norm_layer = norm_layer

        self.filter = nn.Sequential(
            FilteringBlock( 3, 16, norm_layer=norm_layer),
            FilteringBlock(16, 32, norm_layer=norm_layer),
            FilteringBlock(32, 64, norm_layer=norm_layer),
            FilteringBlock(64, 32, norm_layer=norm_layer),
            FilteringBlock(32, 16, norm_layer=norm_layer),
            nn.Conv2d(16, 3, kernel_size=1, stride=1),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.filter(x)
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

        # Clip to [0, 1].
        x = x.clip(0., 1.)
        return x

    @property
    def device(self):
        return next(self.parameters()).device
    

class FilteringBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer='cn'):
        super().__init__()
        assert norm_layer in ['bn', 'cn']
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            ChannelNorm2d(out_channels) if norm_layer == 'cn' else nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.se_layer = SELayer(out_channels, reduction_ratio=8)
        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.se_layer(out)
        if self.proj:
            x = self.proj(x)
        out = out + x
        out = self.relu(out)
        return out


class ChannelNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-3):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = nn.parameter.Parameter(
            torch.empty((num_features,), dtype=torch.float32), requires_grad=True
        )
        self.beta = nn.parameter.Parameter(
            torch.empty((num_features,), dtype=torch.float32), requires_grad=True
        )
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def _get_moments(self, x):
        mean = x.mean(dim=1, keepdim=True)
        variance = torch.sum((x - mean.detach()) ** 2, dim=1, keepdim=True)
        # Divide by N - 1
        variance /= (self.num_features - 1)
        return mean, variance

    def forward(self, x):
        mean, variance = self._get_moments(x)
        x = (x - mean) / (torch.sqrt(variance) + self.eps)
        x = x * self.gamma[None, :, None, None] + self.beta[None, :, None, None]
        return x


class SELayer(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        self.se_module = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_pooled = torch.mean(x, dim=(2, 3), keepdim=False)
        scale = self.se_module(x_pooled)[:, :, None, None]
        out = x * scale
        return out


def build_object_detection_model(cfg):
    cfg = cfg.clone()
    model = GeneralizedRCNN(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    return model