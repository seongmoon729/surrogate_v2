import os
import sys
import argparse


from codec_ops import (
    CODEC_LIST,
    DS_LEVELS,
    JPEG_QUALITIES,
    WEBP_QUALITIES,
    VTM_QUALITIES,
    VVENC_QUALITIES,
)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument(
        "--gpu", "-g", type=str, default="0",
        help="GPU index to use.")
    parser.add_argument(
        "--data_dir", "-d", type=str, default="data",
        help="Path to the dataset.")
    
    subparsers = parser.add_subparsers(
        title="command", dest="command",
        help="Train or Evaluate")
    train_parser = subparsers.add_parser(
        "train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="")
    train_parser.add_argument(
        "--vision_task", "-vt", type=str, default="detection",
        help="Vision task ('classification', 'detection', 'segmentation').")
    train_parser.add_argument(
        "--vision_network", "-vn", type=str, default="faster_rcnn_X_101_32x8d_FPN_3x",
        help="Name of vision task network.")
    train_parser.add_argument(
        "--surrogate_quality", "-sq", type=int, default=1,
        help="Quality of surrogate codec.")
    train_parser.add_argument(
        "--filter_norm_layer", "-fnl", type=str, default='cn',
        help="Normalization layer of filtering network.")
    train_parser.add_argument(
        "--lmbda", "-ld", type=float, default=1.0,
        help="RD trade-off coefficient (R + ld * D).")
    train_parser.add_argument(
        "--batch_size", "-bs", type=int, default=2,
        help="Batch size.")
    train_parser.add_argument(
        "--optimizer", "-opt", type=str, default="adam",
        help="Optimizer.")
    train_parser.add_argument(
        "--learning_rate", "-lr", type=float, default=1e-4,
        help="Learning rate.")
    train_parser.add_argument(
        "--lr_scheduler", "-lrs", type=str, default='constant',
        help="Learning rate scheduler.")
    train_parser.add_argument(
        "--final_lr_rate", "-flrr", type=float, default=1e-1,
        help="Final learning rate factor.")
    train_parser.add_argument(
        "--steps", "-s", type=int, default=50000,
        help="Number of total steps.")
    train_parser.add_argument(
        "--checkpoint_period", "-cp", type=int, default=10000,
        help="Checkpoint period.")
    train_parser.add_argument(
        "--train_downscale", "-td", type=int, default=1,
        help="Image downscale level before the encoding (comma separated).\n"
            f"   * levels: {','.join(map(str, DS_LEVELS))}")
    train_parser.add_argument(
        "--suffix", "-sf", type=str, default="",
        help="Suffix to the session path.")

    eval_parser = subparsers.add_parser(
        "evaluate",
        formatter_class=argparse.RawTextHelpFormatter,
        description="")
    eval_parser.add_argument(
        "--vision_task", "-vt", type=str, default="detection",
        help="Vision task ('classification', 'detection', 'segmentation').")
    eval_parser.add_argument(
        "--vision_network", "-vn", type=str, default="faster_rcnn_X_101_32x8d_FPN_3x",
        help="Name of vision task network.")
    eval_parser.add_argument(
        "--session_path", "-sp", type=str,
        default='out/detection/faster_rcnn_X_101_32x8d_FPN_3x/base',
        help="Saved or baseline path for end-to-end network.\n"
             "Format: 'out/[TASK]/[NETWORK]/[BASE_OR_SAVED_SESSION_PATH]'\n"
             "   ex1) 'out/detection/faster_rcnn_X_101_32x8d_FPN_3x/base'\n"
             "   ex2) 'out/detection/faster_rcnn_X_101_32x8d_FPN_3x/q1_ld1.0/bs2_adam_lr1e-05'")
    eval_parser.add_argument(
        "--session_step", "-ss", type=int, default=0,
        help="Checkpoint step for saved session (0 is dummy for baseline).")
    eval_parser.add_argument(
        "--eval_codec", "-ec", type=str, default='vvenc',
        help="Choose codec to use in evaluation.")
    eval_parser.add_argument(
        "--eval_quality", "-eq", type=str,
        help="Encoding quality (comma separated). There are fixed list of qualities for each codec.\n"
            f"   * jpeg  (2-31): {','.join(map(str, JPEG_QUALITIES)):>17} <- lower is better\n"
            f"   * webp (1-100): {','.join(map(str, WEBP_QUALITIES)):>17} <- higher is better\n"
            f"   * vtm   (0-63): {','.join(map(str, VTM_QUALITIES)):>17} <- lower is better\n"
            f"   * vvenc (0-63): {','.join(map(str, VVENC_QUALITIES)):>17} <- lower is better\n")
    eval_parser.add_argument(
        "--allow_various_qualities", "-avq", action='store_true',
        help="Allow user can evaluate the model with various qualities.")
    eval_parser.add_argument(
        "--eval_downscale", "-ed", type=str, default='0',
        help="Image downscale level before the encoding (comma separated).\n"
            f"   * levels: {','.join(map(str, DS_LEVELS))}")
    eval_parser.add_argument(
        "--coco_classes", "-cc", type=str,
        default='data/open-images-v6-etri/annotations_5k/coco_classes.txt',
        help="Text file for coco classes.")
    eval_parser.add_argument(
        "--input_label_map", "-ilm", type=str,
        default='data/open-images-v6-etri/annotations_5k/coco_label_map.pbtxt',
        help="Open images challenge labelmap.")
    eval_parser.add_argument(
        "--input_annotations_boxes", "-iab", type=str,
        default='data/open-images-v6-etri/annotations_5k/detection_validation_bbox_5k.csv',
        help="File with groundtruth boxes annotations.")
    eval_parser.add_argument(
        "--input_annotations_labels", "-ial", type=str,
        default='data/open-images-v6-etri/annotations_5k/detection_validation_labels_5k.csv',
        help="File with groundtruth labels annotations.")
    eval_parser.add_argument(
        "--segmentation_mask_dir", "-smd", type=str,
        default='data/open-images-v6-etri/annotations_5k/challenge_2019_validation_masks',
        help="Directory to groundtruth segmentation files.")
    eval_parser.add_argument(
        "--input_list", "-il", type=str,
        default='data/open-images-v6-etri/annotations_5k/detection_validation_input_5k.lst',
        help="Text file for input list.")
    eval_parser.add_argument(
        "--input_dir", "-id", type=str,
        default='data/open-images-v6-etri/validation/',
        help="Directory path for inputs.")
    eval_parser.add_argument(
        "--num_parallel_eval_per_gpu", "-npepg", type=int, default=6,
        help="Number of parallel evaluators per gpu.")

    args = parser.parse_args()
    if args.command is None:
        parser.print_usage()
        sys.exit(2)
    if (args.command == 'evaluate' and
        args.eval_codec not in ['none'] and
        not args.allow_various_qualities):
        if args.eval_codec == 'surrogate':
            qualities = [1, 2, 3, 4, 5, 6, 7, 8]
        elif args.eval_codec == 'jpeg':
            qualities = JPEG_QUALITIES
        elif args.eval_codec == 'webp':
            qualities = WEBP_QUALITIES
        elif args.eval_codec == 'vtm':
            qualities = VTM_QUALITIES
        elif args.eval_codec == 'vvenc':
            qualities = VVENC_QUALITIES
        else:
            raise ValueError(f"'{args.eval_codec}' is wrong codec, available: {['none', 'surrogate'] + CODEC_LIST}")
        if args.eval_quality is None:
            assert False, f"Please provide '--eval_quality' (-eq), for {args.eval_codec}, available: {qualities}."

        qs = map(int, args.eval_quality.split(','))
        for q in qs:
            assert q in qualities, f"{q} is not in {qualities}."
        ds = map(int, args.eval_downscale.split(','))
        for d in ds:
            assert d in DS_LEVELS, f"{d} is not in {DS_LEVELS}."
    return args


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['DETECTRON2_DATASETS'] = args.data_dir

    if args.command == 'train':
        import train
        train.train(args)
    else:
        import evaluate
        evaluate.evaluate_for_object_detection(args)


if __name__ == '__main__':
    args = parse_args()
    main(args)