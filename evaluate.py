import warnings
import itertools
from pathlib import Path

import ray
import cv2
import pandas as pd
from tqdm import tqdm

from object_detection.metrics import io_utils
from object_detection.utils import object_detection_evaluation
from object_detection.metrics import oid_challenge_evaluation_utils as oid_utils

import utils
import models

import oid_mask_encoding


class Evaluator:
    def __init__(self, session_path, session_step, coco_classes):
        warnings.filterwarnings('ignore', category=UserWarning)
        self.session_path = session_path
        self.session_step = session_step
        self.coco_classes = coco_classes

        session_path = Path(session_path)
        session_path.mkdir(parents=True, exist_ok=True)    
        (self.vision_task, self.vision_network,
         surrogate_quality, self.is_saved_session) = utils.inspect_session_path(session_path)

        # Build end-to-end network.
        cfg = utils.get_od_cfg(self.vision_task, self.vision_network)
        self.end2end_network = models.EndToEndNetwork(surrogate_quality, self.vision_task, od_cfg=cfg)

        # Restore weights.
        if self.is_saved_session:
            checkpoint = utils.Checkpoint(session_path)
            checkpoint.load(self.end2end_network.filtering_network, step=session_step)

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

        o_lines = []
        for i, coco_cnt_id in enumerate(classes):
            class_name = self.coco_classes[coco_cnt_id]
            result = [imageId, class_name, scores[i]] + bboxes[i].tolist()
            if self.vision_task == 'segmentation':
                result += [
                    masks[i].shape[1],
                    masks[i].shape[0],
                    oid_mask_encoding.encode_binary_mask(masks[i]).decode('ascii')]
            o_line = ','.join(map(str, result))
            o_lines.append(o_line)
        return o_lines, bpp


def evaluate_for_object_detection(config):
    logger = utils.get_logger()
    ray.init()

    session_path = Path(config.session_path)
    session_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Start evaluation script for '{session_path}'.")
    
    vision_task, _, _, _ = utils.inspect_session_path(session_path)

    if ',' in config.eval_downscale:
        eval_downscales = list(map(int, config.eval_downscale.split(',')))
    else:
        eval_downscales = [int(config.eval_downscale)]
    if ',' in config.eval_quality:
        eval_qualities = list(map(int, config.eval_quality.split(',')))
    else:
        eval_qualities = [int(config.eval_quality)]
    eval_settings = list(itertools.product(eval_downscales, eval_qualities))

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
    n_p_eval = config.num_parallel_eval
    eval_builder = ray.remote(num_gpus=(n_gpu / n_p_eval))(Evaluator)
    eval_init_args = (session_path, config.session_step, coco_classes)
    # evaluators = [eval_builder.remote(*eval_init_args) for _ in range(n_p_eval)]

    # Set result/output file names.
    result_path = session_path / 'result.csv'
    # TODO. Remove artifacts.
    coco_output_path = session_path / 'output_coco.csv'
    oid_output_path = session_path / 'output_oid.csv'
    oid_output_eval_path = session_path / 'output_oid_eval.csv'

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
        result_df = pd.DataFrame(columns=['task', 'codec', 'downscale', 'quality', 'bpp', 'metric'])

    #TODO. Memory growth issue (~ 174GB).
    logger.info("Start evaluation loop.")
    total = len(input_files) * len(eval_settings)
    with tqdm(total=total, dynamic_ncols=True, smoothing=0.1) as pbar:
        for downscale, quality in eval_settings:
            input_iter = iter(input_files)
            evaluators = [eval_builder.remote(*eval_init_args) for _ in range(n_p_eval)]
            codec_args = (config.eval_codec, quality, downscale)
            coco_of = open(coco_output_path, 'w')
            bpps = []
            work_info = dict()
            if vision_task == 'detection':
                coco_of.write('ImageID,LabelName,Score,XMin,XMax,YMin,YMax\n')
            else:
                coco_of.write('ImageID,LabelName,Score,XMin,XMax,YMin,YMax,ImageWidth,ImageHeight,Mask\n')
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
                
                # Get results.
                if (not evaluators) or end_flag:
                    done_ids, _ = ray.wait(list(work_info.keys()), timeout=1)
                    if done_ids:
                        for done_id in done_ids:
                            o_lines, bpp = ray.get(done_id)
                            bpps.append(bpp)
                            for o_line in o_lines:
                                coco_of.write(o_line + '\n')
                            eval = work_info.pop(done_id)
                            if end_flag:
                                ray.kill(eval)
                                del eval
                            else:
                                evaluators.append(eval)
                            pbar.update(1)
                # End one loop.
                if not work_info:
                    break
            coco_of.close()

            # Postprocess: Convert coco to oid.
            coco_output_data = pd.read_csv(coco_output_path)
            with open(oid_output_path, 'w') as f:
                f.write(','.join(coco_output_data.columns) + '\n')
                for _, row in coco_output_data.iterrows():
                    coco_id = row['LabelName'].replace(' ', '_')
                    if coco_id in selected_classes:
                        oid_id = coco_id
                        row['LabelName'] = oid_id
                        o_line = ','.join(map(str,row))
                        f.write(o_line + '\n')

            # Open images challenge evaluation.
            if vision_task == 'detection':
                # Generate open image challenge evaluator.
                challenge_evaluator = (
                    object_detection_evaluation.OpenImagesChallengeEvaluator(
                        categories, evaluate_masks=is_instance_segmentation_eval))
                # Load oid result.
                all_predictions = pd.read_csv(oid_output_path)
                # Ready for evaluation.
                for image_id, image_groundtruth in all_annotations.groupby('ImageID'):
                    groundtruth_dictionary = oid_utils.build_groundtruth_dictionary(image_groundtruth, class_label_map)
                    challenge_evaluator.add_single_ground_truth_image_info(image_id, groundtruth_dictionary)
                    prediction_dictionary = oid_utils.build_predictions_dictionary(
                        all_predictions.loc[all_predictions['ImageID'] == image_id], class_label_map)
                    challenge_evaluator.add_single_detected_image_info(image_id, prediction_dictionary)
                # Evaluate.
                metrics = challenge_evaluator.evaluate()
                with open(oid_output_eval_path, 'w') as f:
                    io_utils.write_csv(f, metrics)
                mean_map = sum(metrics.values()) / len(metrics.values())
                mean_bpp = sum(bpps) / len(bpps)
            else:
                pass
            row = {
                'task': vision_task,
                'codec': config.eval_codec,
                'downscale': downscale,
                'quality': quality,
                'bpp': mean_bpp,
                'metric': mean_map,
            }
            result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)
            result_df.sort_values(by=['task', 'codec', 'downscale', 'quality'], inplace=True)
            result_df.to_csv(result_path, index=False)



