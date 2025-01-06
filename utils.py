# Copyright (c) 2024, Ziwen Chen.

import os
import sys
import logging
import functools
from termcolor import colored
import torch
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=''):
    """
    # --------------------------------------------------------
    # Swin Transformer
    # Copyright (c) 2021 Microsoft
    # Licensed under The MIT License [see LICENSE for details]
    # Written by Ze Liu
    # --------------------------------------------------------
    """
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
        colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger

def create_optimizer(model, weight_decay, learning_rate, betas):
    decay_params, nodecay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() == 1 or name.endswith('.bias') or getattr(param, '_no_weight_decay', False):
            nodecay_params.append(param)
        else:
            decay_params.append(param)
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer

def create_scheduler(optimizer, total_train_steps, warm_up_steps, scheduler_type='cosine'):
    if scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, warm_up_steps, total_train_steps)
    elif scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, warm_up_steps, total_train_steps)
    elif scheduler_type == 'constant':
        scheduler = get_constant_schedule_with_warmup(optimizer, warm_up_steps)
    else:
        raise ValueError(f'Invalid scheduler type: {scheduler_type}')
    return scheduler

def auto_resume_helper(checkpoint_dir):
    checkpoints = os.listdir(checkpoint_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('.pt')]
    #print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    resume_file = None
    if len(checkpoints) > 0:
        # rank checkpoints from newest to oldest
        checkpoints = sorted(checkpoints, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
        for i, ckpt in enumerate(checkpoints):
            try:
                checkpoint = torch.load(os.path.join(checkpoint_dir, ckpt), map_location='cpu')
                del checkpoint
            except:
                continue # skip corrupted checkpoint
            resume_file = os.path.join(checkpoint_dir, ckpt)
            break
        #print(f"The latest checkpoint founded: {resume_file}")
    return resume_file