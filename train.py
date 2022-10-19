import os
import sys
import random
from datetime import timedelta

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import profile, schedule, tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
import detectron2.utils.comm as comm
from detectron2.data import build_detection_train_loader

import utils
import models
import checkpoint


_PORT = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
_DIST_URL = f"tcp://127.0.0.1:{_PORT}"
_DEFAULT_TIMEOUT = timedelta(minutes=30)


def train(config):
    n_gpu = len(config.gpu.split(',')) if ',' in config.gpu else 1

    if n_gpu > 1:
        mp.spawn(
            _dist_train_worker,
            nprocs=n_gpu,
            args=(
                _train_for_object_detection,
                n_gpu,
                n_gpu,
                0,
                _DIST_URL,
                (config,),
                _DEFAULT_TIMEOUT,
            ),
            daemon=False,
        )
    else:
        _train_for_object_detection(config)


def _dist_train_worker(
    local_rank,
    main_func,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    dist_url,
    args,
    timeout,
):
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    dist.init_process_group(
        backend="NCCL",
        init_method=dist_url,
        world_size=world_size,
        rank=global_rank,
        timeout=timeout,
    )
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    comm.synchronize()
    main_func(*args)


def _train_for_object_detection(config):
    logger = utils.get_logger()

    # Set session path (path for artifacts of training).
    session_path = 'out' / utils.build_session_path(config)
    if comm.is_main_process():
        logger.info(f"Start training script for '{session_path}'.")

    # Parse lambdas.
    if ',' in config.lmbda:
        lmbda_list = list(map(float, config.lmbda.split(',')))
    else:
        lmbda_list = [float(config.lmbda)]

    # Get detectron2 config data.
    cfg = utils.get_od_cfg(config.vision_task, config.vision_network)

    # Build end-to-end model.
    end2end_network = models.EndToEndNetwork(
        config.surrogate_quality, config.vision_task, config.filter_norm_layer, od_cfg=cfg)

    # Load on GPU.
    end2end_network.cuda()

    # Set mode as training.
    end2end_network.train()

    # Build optimizer.
    target_params = end2end_network.filtering_network.parameters()
    optimizer, lr_scheduler = _create_optimizer(
        target_params,
        config.optimizer,
        config.lr_scheduler,
        config.learning_rate,
        config.steps,
        config.final_lr_rate)

    # Search checkpoint files & resume.
    last_step = 0
    ckpt = checkpoint.Checkpoint(session_path)
    last_step = ckpt.resume(end2end_network.filtering_network, optimizer, lr_scheduler)
    if comm.is_main_process():
        if last_step:
            logger.info(f"Resume training. Last step is {last_step}.")
        else:
            logger.info("Start training from the scratch.")

    # Distributed.
    distributed = comm.get_world_size() > 1
    if distributed:
        end2end_network = DistributedDataParallel(
            end2end_network,
            device_ids=[comm.get_local_rank()],
            broadcast_buffers=False,
            find_unused_parameters=True
        )

    # Create summary writer.
    if comm.is_main_process():
        writer = SummaryWriter(session_path)

    # Build data loader.
    cfg.SOLVER.IMS_PER_BATCH = config.batch_size
    dataloader = build_detection_train_loader(cfg)

    # Run training loop.
    logger.info("Start training.")
    start_step = last_step + 1
    end_step = config.steps

    for data, step in zip(dataloader, range(start_step, end_step + 1)):
        lmbdas = random.choices(lmbda_list, k=(config.batch_size // comm.get_world_size()))
        losses = end2end_network(data, lmbdas)
        loss_rd = losses['r'] + losses['d']
        
        optimizer.zero_grad()
        loss_rd.backward()
        optimizer.step()
        lr_scheduler.step()

        # Calculate reduced losses.
        losses = {k: v.item() for k, v in comm.reduce_dict(losses).items()}
        loss_rd = losses['r'] + losses['d']

        # Write on tensorboard.
        if comm.is_main_process():
            writer.add_scalar('train/loss/rate', losses['r'], step)
            writer.add_scalar('train/loss/distortion', losses['d'], step)
            writer.add_scalar('train/loss/combined', loss_rd, step)
            writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], step)

            if step % 100 == 0:
                logger.info(f"step: {step:6} | loss_r: {losses['r']:7.4f} | loss_d: {losses['d']:7.4f}")
                if distributed:
                    target_network = end2end_network.module.filtering_network
                else:
                    target_network = end2end_network.filtering_network
                ckpt.save(
                    target_network,
                    optimizer,
                    lr_scheduler,
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