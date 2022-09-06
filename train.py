from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from detectron2.data import build_detection_train_loader

import utils
import models
import checkpoint


# TODO. Multi-gpu support.
def train_for_object_detection(config):
    logger = utils.get_logger()

    # Set session path (path for artifacts of training).
    session_path = utils.build_session_path(config)
    logger.info(f"Start training script for '{session_path}'.")

    # Get detectron2 config data.
    cfg = utils.get_od_cfg(config.vision_task, config.vision_network)

    # Build end-to-end model.
    end2end_network = models.EndToEndNetwork(
        config.surrogate_quality, config.vision_task, od_cfg=cfg)

    # Build optimizer.
    target_params = (
        list(end2end_network.filtering_network.filter.parameters())
        + list(end2end_network.filtering_network.pixel_rate_estimator.parameters()))
    optimizer, lr_scheduler = _create_optimizer(
        target_params,
        config.optimizer,
        config.lr_scheduler,
        config.learning_rate,
        config.steps,
        config.final_lr_rate)

    # Search checkpoint files & resume.
    output_path = Path('out') / session_path
    ckpt = checkpoint.Checkpoint(output_path)
    last_step = ckpt.resume(end2end_network.filtering_network, optimizer)
    if last_step:
        logger.info(f"Resume training. Last step is {last_step}.")
    else:
        logger.info("Start training from the scratch.")

    # Create summary writer.
    writer = SummaryWriter(output_path)

    # Set as training mode & load on GPU.
    end2end_network.train()
    end2end_network.cuda()
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    # Build data loader.
    cfg.SOLVER.IMS_PER_BATCH = config.batch_size
    dataloader = build_detection_train_loader(cfg)

    # Run training loop.
    logger.info("Start training.")
    start_step = last_step + 1
    end_step = config.steps
    for data, step in zip(dataloader, range(start_step, end_step + 1)):
        losses = end2end_network(data)
        loss_rd = losses['r'] + config.lmbda * losses['d']
        
        optimizer.zero_grad()
        loss_rd.backward()
        optimizer.step()
        lr_scheduler.step()

        writer.add_scalar('Loss/rate', losses['r'].item(), step)
        writer.add_scalar('Loss/distortion', losses['d'].item(), step)
        writer.add_scalar('Loss/combined', loss_rd.item(), step)
        writer.add_scalar('lr', lr_scheduler.get_last_lr()[0])

        if step % 100 == 0:
            logger.info(f"step: {step:6} | loss_r: {losses['r']:7.4f} | loss_d: {losses['d']:7.4f}")        
            ckpt.save(
                end2end_network.filtering_network,
                optimizer,
                step=step,
                persistent_period=config.checkpoint_period)


def _create_optimizer(parameters, optim_name, scheduler_name, initial_lr, total_steps, final_rate=.1):
    if optim_name == 'sgd':
        optimizer = optim.SGD(parameters, lr=initial_lr, momentum=0.9)
    elif optim_name == 'adam':
        optimizer = optim.Adam(parameters, lr=initial_lr)
    else:
        raise NotImplemented("Currently supported optimizers are 'adam' and 'sgd'.")

    if scheduler_name == 'constant':
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    elif scheduler_name == 'exponential':
        gamma = final_rate ** (1 / total_steps)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:
        raise NotImplemented("Currently supported schedulers are 'constant' and 'exponential'.")
    
    return optimizer, scheduler