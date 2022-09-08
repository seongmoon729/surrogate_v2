import warnings
import itertools
from pathlib import Path

import ray
import cv2
import pandas as pd
from tqdm import tqdm

from object_detection.utils import object_detection_evaluation
from object_detection.metrics import oid_challenge_evaluation_utils as oid_utils

import utils
import models
import checkpoint
import oid_mask_encoding


class Evaluator:
    def __init__(self, session_path, session_step, coco_classes):
        warnings.filterwarnings('ignore', category=UserWarning)
        self.session_path = session_path
        self.session_step = session_step
        self.coco_classes = coco_classes

        session_path = Path(session_path)
        session_path.mkdir(parents=True, exist_ok=True)    
        (self.vision_task,
         self.vision_network,
         surrogate_quality,
         self.is_saved_session) = utils.inspect_session_path(session_path)

        # Build end-to-end network.
        cfg = utils.get_od_cfg(self.vision_task, self.vision_network)
        self.end2end_network = models.EndToEndNetwork(
            surrogate_quality, self.vision_task, od_cfg=cfg)

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
                od_output += [
                    masks[i].shape[1],
                    masks[i].shape[0],
                    oid_mask_encoding.encode_binary_mask(masks[i]).decode('ascii')]
            od_outputs.append(od_output)
        return od_outputs, bpp


def evaluate_for_object_detection(config):
    logger = utils.get_logger()
    logger.info(f"Start evaluation script for '{config.session_path}'.")
    ray.init()

    session_path = Path(config.session_path)
    session_path.mkdir(parents=True, exist_ok=True)
    result_path = session_path / 'result.csv'
    
    # Parse vision task.
    vision_task, _, _, _ = utils.inspect_session_path(session_path)

    # Generate evaluation settings.
    if config.eval_codec == 'surrogate':
        eval_settings = [(None, None)]
    else:
        if ',' in config.eval_downscale:
            eval_downscales = list(map(int, config.eval_downscale.split(',')))
        else:
            eval_downscales = [int(config.eval_downscale)]
        if ',' in config.eval_quality:
            eval_qualities = list(map(int, config.eval_quality.split(',')))
        else:
            eval_qualities = [int(config.eval_quality)]
        eval_settings = list(itertools.product(eval_downscales, eval_qualities))

    # Create or load result dataframe.
    if result_path.exists():
        result_df = pd.read_csv(result_path)
        # Delete already evaluated settings.
        subset_df = result_df[result_df.task == vision_task]
        subset_df = subset_df[subset_df.codec == config.eval_codec]
        evaluated_settings = itertools.product(subset_df.downscale, subset_df.quality)
        for _setting in evaluated_settings:
            if _setting in eval_settings:
                eval_settings.remove(_setting)
    else:
        result_df = pd.DataFrame(
            columns=['task', 'codec', 'downscale', 'quality', 'bpp', 'metric', 'step'])

    input_files = utils.get_input_files(config.input_list, config.input_dir)
    logger.info(f"Number of total images: {len(input_files)}")

    # Read coco classes file.
    coco_classes = open(config.coco_classes, 'r').read().splitlines()

    # Read input label map.
    class_label_map, categories = utils.read_label_map(config.input_label_map)
    selected_classes = list(class_label_map.keys())

    # Read annotation files.
    all_location_annotations = pd.read_csv(config.input_annotations_boxes)
    all_label_annotations = pd.read_csv(config.input_annotations_labels)
    all_label_annotations.rename(columns={'Confidence': 'ConfidenceImageLabel'}, inplace=True)
    is_instance_segmentation_eval = False
    # TODO. Segmentation.
    all_annotations = pd.concat([all_location_annotations, all_label_annotations])

    # Create evaluators.
    n_gpu = len(config.gpu.split(',')) if ',' in config.gpu else 1
    n_eval = config.num_parallel_eval_per_gpu * n_gpu
    eval_builder = ray.remote(num_gpus=(1 / config.num_parallel_eval_per_gpu))(Evaluator)
    eval_init_args = (session_path, config.session_step, coco_classes)

    #TODO. Memory growth issue (~ 174GB).
    logger.info("Start evaluation loop.")
    total = len(input_files) * len(eval_settings)
    with tqdm(total=total, dynamic_ncols=True, smoothing=0.1) as pbar:
        for downscale, quality in eval_settings:
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
            if vision_task == 'segmentation':
                columns += ['ImageWidth', 'ImageHeight', 'Mask']
            od_output_df = pd.DataFrame(od_outputs, columns=columns)

            # Fix & filter the image label.
            od_output_df['LabelName'] = od_output_df['LabelName'].replace(' ', '_', regex=True)
            od_output_df = od_output_df[od_output_df['LabelName'].isin(selected_classes)]

            # Open images challenge evaluation.
            if vision_task == 'detection':
                # Generate open image challenge evaluator.
                challenge_evaluator = (
                    object_detection_evaluation.OpenImagesChallengeEvaluator(
                        categories, evaluate_masks=is_instance_segmentation_eval))
                # Ready for evaluation.
                for image_id, image_groundtruth in all_annotations.groupby('ImageID'):
                    groundtruth_dictionary = oid_utils.build_groundtruth_dictionary(image_groundtruth, class_label_map)
                    challenge_evaluator.add_single_ground_truth_image_info(image_id, groundtruth_dictionary)
                    prediction_dictionary = oid_utils.build_predictions_dictionary(
                        od_output_df.loc[od_output_df['ImageID'] == image_id], class_label_map)
                    challenge_evaluator.add_single_detected_image_info(image_id, prediction_dictionary)

                # Evaluate. class-wise evaluation result is produced.
                metrics = challenge_evaluator.evaluate()
                mean_map = list(metrics.values())[0]
                mean_bpp = sum(bpps) / len(bpps)
            else:
                pass
            result = {
                'task'     : vision_task,
                'codec'    : config.eval_codec,
                'downscale': downscale,
                'quality'  : quality,
                'bpp'      : mean_bpp,
                'metric'   : mean_map,
                'step'     : config.session_step,
            }
            result_df = pd.concat([result_df, pd.DataFrame([result])], ignore_index=True)
            result_df.sort_values(
                by=['task', 'codec', 'downscale', 'bpp'], inplace=True)
            result_df.to_csv(result_path, index=False)



