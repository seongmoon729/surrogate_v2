import warnings
import itertools
from pathlib import Path

import ray
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

from object_detection.utils import object_detection_evaluation
from object_detection.metrics import oid_challenge_evaluation_utils as oid_utils

import utils
import models
import checkpoint
import oid_mask_encoding


class Evaluator:
    def __init__(self, vision_task, vision_network, session_path, session_step, coco_classes, surrogate_quality=None):
        warnings.filterwarnings('ignore', category=UserWarning)
        self.vision_task = vision_task
        self.vision_network = vision_network
        self.session_path = session_path
        self.session_step = session_step
        self.coco_classes = coco_classes

        session_path = Path(session_path)
        session_path.mkdir(parents=True, exist_ok=True)
        if surrogate_quality:
            _, self.is_saved_session, norm_layer = utils.inspect_session_path(session_path)
        else:
            surrogate_quality, self.is_saved_session, norm_layer = utils.inspect_session_path(session_path)

        # dummy
        if surrogate_quality is None:
            surrogate_quality = 1

        # Build end-to-end network.
        cfg = utils.get_od_cfg(vision_task, vision_network)
        self.end2end_network = models.EndToEndNetwork(
            surrogate_quality, vision_task, norm_layer, od_cfg=cfg)

        # Restore weights.
        if self.is_saved_session:
            ckpt = checkpoint.Checkpoint(session_path)
            ckpt.load(self.end2end_network.filtering_network, step=session_step)

        # Set evaluation mode & load on GPU.
        self.end2end_network.eval()
        self.end2end_network.cuda()
    
    def step(self, input_file, codec, quality, downscale):
        image = cv2.imread(str(input_file))
        outputs = self.end2end_network(
            image,
            eval_codec=codec,
            eval_quality=quality,
            eval_downscale=downscale,
            eval_filtering=self.is_saved_session)
        imageId = input_file.stem
        classes = outputs['instances'].pred_classes.to('cpu').numpy()
        scores = outputs['instances'].scores.to('cpu').numpy()
        bboxes = outputs['instances'].pred_boxes.tensor.to('cpu').numpy()
        H, W = outputs['instances'].image_size

        # Bit per pixel.
        bpp = outputs['bpp']

        # convert bboxes to 0-1
        bboxes = bboxes / [W, H, W, H]

        # OpenImage output x1, x2, y1, y2 in percentage
        bboxes = bboxes[:, [0, 2, 1, 3]]

        if self.vision_task == 'segmentation':
            masks = outputs['instances'].pred_masks.to('cpu').numpy()

        od_outputs = []
        for i, coco_cnt_id in enumerate(classes):
            class_name = self.coco_classes[coco_cnt_id]
            od_output = [imageId, class_name, scores[i]] + bboxes[i].tolist()
            if self.vision_task == 'segmentation':
                assert all(map(lambda a, b: a == b, masks[i].shape[:2], [H, W])), \
                    f"Size of resulting mask does not match the input size: {imageId}"
                od_output += [
                    masks[i].shape[1],
                    masks[i].shape[0],
                    oid_mask_encoding.encode_binary_mask(masks[i]).decode('ascii')]
            od_outputs.append(od_output)
        return od_outputs, bpp


def evaluate_for_object_detection(config):
    logger = utils.get_logger()
    logger.info(f"Evaluate '{config.session_path}' session with step size {config.session_step}.")
    logger.info(f"Task    : {config.vision_task}")
    logger.info(f"Network : {config.vision_network}")
    logger.info(f"Codec   : {config.eval_codec}")
    ray.init()

    session_path = Path(config.session_path)
    session_path.mkdir(parents=True, exist_ok=True)
    result_path = session_path / 'result.csv'

    # Generate evaluation settings.
    if ',' in config.eval_downscale:
        eval_downscales = list(map(int, config.eval_downscale.split(',')))
    else:
        eval_downscales = [int(config.eval_downscale)]
    
    if config.eval_quality:
        if ',' in config.eval_quality:
            eval_qualities = list(map(int, config.eval_quality.split(',')))
        else:
            eval_qualities = [int(config.eval_quality)]
    else:
        eval_qualities = [None]
    eval_settings = list(itertools.product(eval_downscales, eval_qualities))

    # Create or load result dataframe.
    if result_path.exists():
        result_df = pd.read_csv(result_path)
        subset_df = result_df.copy()
        # Delete already evaluated settings.
        subset_df = subset_df[subset_df.step == config.session_step]
        subset_df = subset_df[subset_df.task == config.vision_task]
        subset_df = subset_df[subset_df.codec == config.eval_codec]
        evaluated_settings = itertools.product(subset_df.downscale, subset_df.quality)
        for _setting in evaluated_settings:
            if _setting in eval_settings:
                eval_settings.remove(_setting)
    else:
        result_df = pd.DataFrame(
            columns=['task', 'codec', 'downscale', 'quality', 'bpp', 'metric', 'step'])

     # Set path of input files.
    input_base_dir     = Path(config.input_dir)
    input_img_dir      = input_base_dir / 'validation'
    annot_dir          = input_base_dir / 'annotations_5k'
    coco_class_file    = annot_dir / 'coco_classes.txt'
    coco_labelmap_file = annot_dir / 'coco_label_map.pbtxt'
    input_list_file    = annot_dir / f"{config.vision_task}_validation_input_5k.lst"
    input_annot_boxes  = annot_dir / f"{config.vision_task}_validation_bbox_5k.csv"
    input_annot_labels = annot_dir / f"{config.vision_task}_validation_labels_5k.csv"

    if config.vision_task == 'segmentation':
        gt_segm_mask_dir  = annot_dir / 'challenge_2019_validation_masks'
        input_annot_masks = annot_dir / f"{config.vision_task}_validation_masks_5k.csv"

    input_files = utils.get_input_files(input_list_file, input_img_dir)
    logger.info(f"Number of total images: {len(input_files)}")

    # Create evaluators.
    n_gpu = len(config.gpu.split(',')) if ',' in config.gpu else 1
    n_eval = config.num_parallel_eval_per_gpu * n_gpu
    eval_builder = ray.remote(num_gpus=(1 / config.num_parallel_eval_per_gpu))(Evaluator)

    #TODO. Memory growth issue (~ 174GB).
    logger.info("Start evaluation loop.")
    total = len(input_files) * len(eval_settings)
    with tqdm(total=total, dynamic_ncols=True, smoothing=0.1) as pbar:
        for downscale, quality in eval_settings:

            # Read coco classes file.
            coco_classes = open(coco_class_file, 'r').read().splitlines()

            # Read input label map.
            class_label_map, categories = utils.read_label_map(coco_labelmap_file)
            selected_classes = list(class_label_map.keys())

            # Read annotation files.
            all_location_annotations = pd.read_csv(input_annot_boxes)
            all_label_annotations = pd.read_csv(input_annot_labels)
            all_label_annotations.rename(columns={'Confidence': 'ConfidenceImageLabel'}, inplace=True)
            is_instance_segmentation_eval = False
            if config.vision_task == 'segmentation':
                anno_gt = pd.read_csv(input_annot_masks)
                is_instance_segmentation_eval = True

            eval_init_args = (
                config.vision_task,
                config.vision_network,
                session_path,
                config.session_step,
                coco_classes)
            if config.eval_codec == 'surrogate':
                eval_init_args += (quality,)
                
            # Make/set evaluators and their inputs.
            evaluators = [eval_builder.remote(*eval_init_args) for _ in range(n_eval)]
            input_iter = iter(input_files)
            codec_args = (config.eval_codec, quality, downscale)
            # Run evaluators.
            od_outputs, bpps = [], []
            work_info = dict()
            while True:
                # Put inputs.
                try:
                    if evaluators:
                        file = next(input_iter)
                        eval = evaluators.pop()
                        work_id = eval.step.remote(file, *codec_args)
                        work_info.update({work_id: eval})
                        end_flag = False
                except StopIteration:
                    end_flag = True
                
                # Get detection result & bpp.
                if (not evaluators) or end_flag:
                    done_ids, _ = ray.wait(list(work_info.keys()), timeout=1)
                    if done_ids:
                        for done_id in done_ids:
                            # Store outputs.
                            od_outputs_, bpp = ray.get(done_id)
                            od_outputs.extend(od_outputs_)
                            bpps.append(bpp)
                            eval = work_info.pop(done_id)
                            if end_flag:
                                ray.kill(eval)
                                del eval
                            else:
                                evaluators.append(eval)
                            pbar.update(1)
                # End loop for one setting.
                if not work_info:
                    break

            # Postprocess: Convert coco to oid.
            columns = ['ImageID', 'LabelName', 'Score', 'XMin', 'XMax', 'YMin', 'YMax']
            if config.vision_task == 'segmentation':
                columns += ['ImageWidth', 'ImageHeight', 'Mask']
            od_output_df = pd.DataFrame(od_outputs, columns=columns)

            # Fix & filter the image label.
            od_output_df['LabelName'] = od_output_df['LabelName'].replace(' ', '_', regex=True)
            od_output_df = od_output_df[od_output_df['LabelName'].isin(selected_classes)]

            # Resize GT segmentation labels.
            if config.vision_task == 'segmentation':
                all_segm_annotations = pd.read_csv(input_annot_masks)
                for idx, row in anno_gt.iterrows():
                    pred_rslt = od_output_df.loc[od_output_df['ImageID'] == row['ImageID']]
                    if not len(pred_rslt):
                        logger.info(f"Image not in prediction: {row['ImageID']}")
                        continue
                    
                    W, H = pred_rslt['ImageWidth'].iloc[0], pred_rslt['ImageHeight'].iloc[0]

                    mask_img = Image.open(gt_segm_mask_dir / row['MaskPath'])

                    if any(map(lambda a, b: a != b, mask_img.size, [W, H])):
                        mask_img = mask_img.resize((W, H))
                        mask = np.asarray(mask_img)
                        mask_str = oid_mask_encoding.encode_binary_mask(mask).decode('ascii')
                        all_segm_annotations.at[idx, 'Mask'] = mask_str
                        all_segm_annotations.at[idx, 'ImageWidth'] = W
                        all_segm_annotations.at[idx, 'ImageHeight'] = H

                all_location_annotations = oid_utils.merge_boxes_and_masks(
                    all_location_annotations, all_segm_annotations)
            
            all_annotations = pd.concat([all_location_annotations, all_label_annotations])

            # Open images challenge evaluation.
            # Generate open image challenge evaluator.
            challenge_evaluator = (
                object_detection_evaluation.OpenImagesChallengeEvaluator(
                    categories, evaluate_masks=is_instance_segmentation_eval))
            # Ready for evaluation.

            # for image_id, image_groundtruth in all_annotations.groupby('ImageID'):
            with tqdm(all_annotations.groupby('ImageID')) as tbar:
                for image_id, image_groundtruth in tbar:
                    groundtruth_dictionary = oid_utils.build_groundtruth_dictionary(image_groundtruth, class_label_map)
                    challenge_evaluator.add_single_ground_truth_image_info(image_id, groundtruth_dictionary)
                    prediction_dictionary = oid_utils.build_predictions_dictionary(
                        od_output_df.loc[od_output_df['ImageID'] == image_id], class_label_map)
                    challenge_evaluator.add_single_detected_image_info(image_id, prediction_dictionary)

            # Evaluate. class-wise evaluation result is produced.
            metrics = challenge_evaluator.evaluate()
            mean_map = list(metrics.values())[0]
            mean_bpp = sum(bpps) / len(bpps)
                
            result = {
                'task'     : config.vision_task,
                'codec'    : config.eval_codec,
                'downscale': downscale,
                'quality'  : quality,
                'bpp'      : mean_bpp,
                'metric'   : mean_map,
                'step'     : config.session_step,
            }
            result_df = pd.concat([result_df, pd.DataFrame([result])], ignore_index=True)
            result_df.sort_values(
                by=['task', 'codec', 'downscale', 'step', 'bpp'], inplace=True)
            result_df.to_csv(result_path, index=False)



