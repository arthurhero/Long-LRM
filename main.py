# Copyright (c) 2024, Ziwen Chen.

import os
import argparse
from easydict import EasyDict as edict
import wandb
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from data.dataset import Dataset
from model.llrm import LongLRM

# DDP setup


# config setup


# dataloader


# model setup


# optimizer, scheduler, checkpoint and amp


# wandb setup


# evaluation


# training


