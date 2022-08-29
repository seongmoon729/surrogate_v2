import logging
from pathlib import Path

import torch
import torch.optim as optim

from detectron2 import model_zoo
from detectron2 import config as detectron2_config
from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2


class Checkpoint:
    def __init__(self, path):
        self.base_path = Path(path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, network, optimizer=None, step=0, persistent_period=None):
        ckpt_path = self.base_path / f"checkpoint_{step}.pth"

        # Choose variables to save.
        target_objects = {'network': network.state_dict(), 'step': step}
        if optimizer:
            target_objects.update({'optimizer': optimizer.state_dict()})

        # Save.
        torch.save(target_objects, ckpt_path)
        
        # Delete old checkpoints except persistent checkpoints.
        old_ckpt_paths = set(self.base_path.glob('*.pth')) - set([ckpt_path])
        for old_ckpt_path in old_ckpt_paths:
            if self._get_step(old_ckpt_path) % persistent_period:
                old_ckpt_path.unlink()

    def load(self, network, optimizer=None, step=-1):
        ckpt_paths = sorted(self.base_path.glob('*.pth'), key=self._get_step)

        if len(ckpt_paths):
            if step == -1:
                target_ckpt_path = ckpt_paths[step]
            else:
                target_ckpt_path = self.base_path / f"checkpoint_{step}.pth"
                assert target_ckpt_path.exists()
            
            # Load.
            target_ckpt = torch.load(target_ckpt_path, map_location=torch.device('cpu'))
            network.load_state_dict(target_ckpt['network'])
            if optimizer:
                optimizer.load_state_dict(target_ckpt['optimizer'])
            step = target_ckpt['step']
        else:
            step = 0
        return step

    def resume(self, network, optimizer, step=-1):
        step = self.load(network, optimizer, step)
        return step

    def _get_step(self, ckpt_path):
        step = int(ckpt_path.stem.split('_')[-1])
        return step


def get_logger():
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def build_session_path(config):
    session_path = Path(config.vision_task)
    session_path /= config.vision_network
    session_path /= (
        f"q{config.surrogate_quality}_ld{config.lmbda}_{config.suffix}" if config.suffix
        else f"q{config.surrogate_quality}_ld{config.lmbda}"
    )
    session_path /= f"bs{config.batch_size}_{config.optimizer}_lr{config.learning_rate}"
    return session_path


def inspect_session_path(session_path):
    session_path = Path(session_path)
    is_saved_session = 'base' not in session_path.name

    vision_network = session_path.parent
    vision_task = session_path.parent.parent
    surrogate_quality = 1  # dummy
    if is_saved_session:
        vision_network = vision_network.parent
        vision_task = vision_task.parent
        surrogate_quality = int(session_path.parent.name.split('_')[0][1:])
    vision_network = vision_network.name
    vision_task = vision_task.name
    return vision_task, vision_network, surrogate_quality, is_saved_session


def get_od_cfg(od_task, od_network):
    if od_task == 'detection':
        config_path = 'COCO-Detection'
    elif od_task == 'segmentation':
        config_path = 'COCO-InstanceSegmentation'
    config_file = config_path + '/' + od_network + '.yaml'
    cfg = detectron2_config.get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return cfg


def get_input_files(input_list, input_dir):
    input_dir = Path(input_dir)
    input_candidates = list(input_dir.glob('*'))
    input_files = open(input_list, 'r').readlines()
    input_files = set(map(lambda x: x.strip(), input_files))
    if len(set(map(lambda x: x.name, input_candidates)) & input_files) == 0:
        ids_candidate = set(map(lambda x: x.stem, input_candidates))
        ids_inputs = set(map(lambda x: x[:-4], input_files))
        if ids_candidate & ids_inputs:  # '.png' -> '.jpg'
            input_files = map(lambda x: x[:-4] + '.jpg', input_files)
        else:
            assert False, "No file is matched."
    input_files = list(map(lambda x: input_dir / x, input_files))
    return input_files


def read_label_map(path):
    label_map = string_int_label_map_pb2.StringIntLabelMap()
    with open(path, 'r') as f:
        label_map_str = f.read()
    text_format.Merge(label_map_str, label_map)
    label_map_dict = dict()
    categories = []
    for item in label_map.item:
        label_map_dict[item.name] = item.id
        categories.append({'id': item.id, 'name': item.name})
    return label_map_dict, categories



def create_optimizer(optim_name, learning_rate, parameters):
    if optim_name == 'sgd':
        optimizer = optim.SGD(parameters, lr=learning_rate, momentum=0.9)
    elif optim_name == 'adam':
        optimizer = optim.Adam(parameters, lr=learning_rate)
    else:
        raise NotImplemented("Currently supported optimizers are 'adam' and 'sgd'.")
    return optimizer