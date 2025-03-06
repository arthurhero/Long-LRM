# Copyright (c) 2024, Ziwen Chen.

import os
import shutil
import copy
import argparse
from easydict import EasyDict as edict
import wandb
import yaml
import random
import time
import datetime
import numpy as np
import cv2
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision

from utils import create_logger, create_optimizer, create_scheduler, auto_resume_helper
from data.dataset import Dataset
from model.llrm import LongLRM

# config setup
parser = argparse.ArgumentParser(description='Long-LRM arguments')
parser.add_argument('--config', type=str, required=True, help='path to config file')
parser.add_argument('--default-config', type=str, help='path to default config file')
parser.add_argument('--evaluation', action='store_true', help='evaluation mode')
args = parser.parse_args()
config_name = os.path.basename(args.config).split('.')[0]
config = yaml.safe_load(open(args.config, 'r'))
def recursive_merge(dict1, dict2):
    for key, value in dict2.items():
        if key not in dict1:
            dict1[key] = value
        elif isinstance(value, dict):
            dict1[key] = recursive_merge(dict1[key], value)
        else:
            dict1[key] = value
    return dict1
if args.default_config is not None:
    default_config = yaml.safe_load(open(args.default_config, 'r'))
    default_config = recursive_merge(default_config, config)
    config = default_config
config_dict = config
config = edict(config)
if args.evaluation:
    config.evaluation = True
if config.get("config_name", None) is not None:
    config_name = config.config_name
checkpoint_dir = os.path.join(config.checkpoint_dir, config_name)
os.makedirs(checkpoint_dir, exist_ok=True)

# torch and DDP setup
rank = int(os.environ["RANK"])
world_size = int(os.environ['WORLD_SIZE'])
local_rank = int(os.environ['LOCAL_RANK'])
local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
group_rank = int(os.environ['GROUP_RANK'])
device = "cuda:{}".format(local_rank)
torch.cuda.set_device(device)
torch.cuda.empty_cache()
seed = 1111 + rank
torch.manual_seed(seed)
torch.random.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.distributed.init_process_group(backend='nccl')
torch.distributed.barrier()

torch.backends.cuda.matmul.allow_tf32 = config.use_tf32
torch.backends.cudnn.allow_tf32 = config.use_tf32

logger_dir = os.path.join(checkpoint_dir, 'logs')
os.makedirs(logger_dir, exist_ok=True)
logger = create_logger(output_dir=logger_dir, dist_rank=rank, name=config_name)
logger.info(f"Rank {rank} / {world_size} with local rank {local_rank} / {local_world_size} and group rank {group_rank}")
logger.info("Config:\n"+yaml.dump(config_dict, sort_keys=False))

torch.distributed.barrier()

# dataloader
dataset = Dataset(config)
if rank == 0:
    data_example = dataset[0]
    os.makedirs(os.path.join(checkpoint_dir, 'data_example'), exist_ok=True)
    data_desc = ""
    for key, value in data_example.items():
        if isinstance(value, torch.Tensor):
            data_desc += "data key: {}, shape: {}\n".format(key, value.size())
        else:
            data_desc += "data key: {}, value: {}\n".format(key, value)
        if key == 'input_images':
            input_images = value # (V, C, H, W)
            input_images = input_images.permute(1, 2, 0, 3).flatten(2, 3) # (C, H, V*W)
            torchvision.utils.save_image(input_images, os.path.join(checkpoint_dir, 'data_example', 'input_images.png'))
    logger.info("Data example:\n"+data_desc)
torch.distributed.barrier()

datasampler = DistributedSampler(dataset, shuffle=not config.get("evaluation", False))
batch_size_per_gpu = config.training.batch_size_per_gpu
dataloader = DataLoader(dataset, 
                        batch_size=batch_size_per_gpu,
                        shuffle=False,
                        num_workers=config.training.num_workers,
                        persistent_workers=True,
                        pin_memory=False,
                        drop_last=True,
                        prefetch_factor=config.training.prefetch_factor,
                        sampler=datasampler)

# model setup
model = LongLRM(config, device).to(device)
model = DDP(model, device_ids=[local_rank])
enable_grad_scaler = config.use_amp and config.amp_dtype == "fp16"
scaler = torch.cuda.amp.GradScaler(enabled=enable_grad_scaler)
amp_dtype_mapping = {"fp16": torch.float16, "bf16": torch.bfloat16}

# optimizer, scheduler, load checkpoint
if not config.get("evaluation", False):
    train_steps = config.training.train_steps
    grad_accum_steps = config.training.grad_accum_steps
    param_update_steps = train_steps
    train_steps = train_steps * grad_accum_steps
    total_batch_size = batch_size_per_gpu * world_size * grad_accum_steps
    num_epochs = int(param_update_steps * total_batch_size / len(dataset))
    logger.info(f"train_steps: {train_steps}, grad_accum_steps: {grad_accum_steps}, param_update_steps: {param_update_steps}, batch_size_per_gpu: {batch_size_per_gpu}, world_size: {world_size}, batch_size_total: {total_batch_size}, dataset_size: {len(dataset)}, num_epochs: {num_epochs}")
    optimizer = create_optimizer(model, config.training.weight_decay, config.training.lr,
                                (config.training.beta1, config.training.beta2))
    scheduler = create_scheduler(optimizer, param_update_steps, config.training.warmup_steps,
                                 config.training.get('scheduler_type', 'cosine'))
num_params = sum(p.numel() for p in model.parameters())
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Number of parameters: {num_params / 1e6:.2f}M, trainable: {num_trainable_params / 1e6:.2f}M")
train_steps_done = 0
resume_file = auto_resume_helper(checkpoint_dir)
auto_resume = resume_file is not None
if resume_file is None:
    resume_file = config.training.get('resume_ckpt', None)
if resume_file is None:
    logger.info("No checkpoint founded, start from scratch")
else:
    logger.info(f"Resume from checkpoint: {resume_file}")
    checkpoint = torch.load(resume_file, map_location=device)
    if isinstance(model, DDP):
        status = model.module.load_state_dict(checkpoint['model'], strict=False)
    else:
        status = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(f"Loaded model with status: {status}")

    if (not config.get("evaluation", False)) and (auto_resume or (not config.training.get("reset_training_state", False))): 
        train_steps_done = checkpoint['train_steps_done']
        logger.info(f"Resume from train_steps_done: {train_steps_done}, param_update_steps_done: {train_steps_done // grad_accum_steps}")
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info(f"Loaded optimizer and scheduler")
        except:
            logger.warning(f"Failed to load optimizer and scheduler, start with new optimizer and scheduler")

# wandb setup
if rank == 0  and not config.get("evaluation", False):
    api_key_path = config.get("api_key_path", None)
    if api_key_path is None or (not os.path.exists(api_key_path)):
        logger.error(f"API key file does not exist: {api_key_path}")
        raise FileNotFoundError(f"API key file does not exist: {api_key_path}")
    api_keys = edict(yaml.safe_load(open(api_key_path, 'r')))
    wandb_api_key = api_keys.wandb
    os.environ["WANDB_API_KEY"] = api_keys.wandb
    if config.training.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    wandb.login()
    if train_steps_done > 0:
        try:
            id_file = open(os.path.join(checkpoint_dir, "wandb_id.txt"), "r")
            wandb_id = id_file.read().strip()
        except:
            wandb_id = wandb.util.generate_id()
            with open(os.path.join(checkpoint_dir, "wandb_id.txt"), "w") as f:
                f.write(wandb_id)
    else:
        wandb_id = wandb.util.generate_id()
        with open(os.path.join(checkpoint_dir, "wandb_id.txt"), "w") as f:
            f.write(wandb_id)
    wandb_dir_path = "/home/user/wandb"
    os.makedirs(wandb_dir_path, exist_ok=True)
    wandb.init(entity=config.training.wandb_entity,
               project=config.training.wandb_project,
               name=config_name,
               id=wandb_id,
               dir=wandb_dir_path,
               config=copy.deepcopy(config),
               resume="allow")
    wandb.run.log_code(".")
    logger.info(f"Initialized wandb")

# evaluation
if config.get("evaluation", False):
    evaluation_dir = os.path.join(config.evaluation_dir, config_name)
    if rank == 0:
        from lpips import LPIPS
        lpips_fn = LPIPS(net="vgg")
        del lpips_fn
        if os.path.exists(evaluation_dir):
            shutil.rmtree(evaluation_dir)
        os.makedirs(evaluation_dir, exist_ok=True)
    torch.distributed.barrier()
    datasampler.set_epoch(0)
    model.eval()
    with model.no_sync(), torch.no_grad(), torch.autocast(
        enabled=config.use_amp,
        device_type="cuda",
        dtype=amp_dtype_mapping[config.amp_dtype],
    ):
        for data in dataloader:
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device)
            ret_dict = model(data)
            model.module.save_evaluation_results(data, ret_dict, evaluation_dir)
        torch.cuda.empty_cache()
    torch.distributed.barrier()
    # aggregate results
    if rank == 0:
        scene_folders = os.listdir(evaluation_dir)
        logger.info(f"Found evaluation results for {len(scene_folders)} scenes")
        metric_dict = {}
        for scene_folder in scene_folders:
            scene_dir = os.path.join(evaluation_dir, scene_folder)
            with open(os.path.join(scene_dir, "metrics.csv"), "r") as f:
                metric_names = f.readline().strip().split(",")[1:]
                metric_lines = f.readlines()[-3:]
                scene_metrics = metric_lines[0].strip().split(",")[1:]
                scene_metrics = [float(x) for x in scene_metrics]
                for key, value in zip(metric_names, scene_metrics):
                    if key not in metric_dict:
                        metric_dict[key] = []
                    metric_dict[key].append(value)
                for line in metric_lines[1:]:
                    key, value = line.strip().split(",")
                    if key not in metric_dict:
                        metric_dict[key] = []
                    metric_dict[key].append(float(value))
        with open(os.path.join(evaluation_dir, "summary.csv"), "w") as f:
            f.write("scene_name,"+",".join(metric_dict.keys())+"\n")
            for i, scene_folder in enumerate(scene_folders):
                f.write(scene_folder+","+",".join([str(metric_dict[key][i]) for key in metric_dict.keys()])+"\n")
            eval_res = ""
            for key in metric_dict.keys():
                scene_num = len(metric_dict[key])
                metric_dict[key] = np.mean(metric_dict[key])
                eval_res += f"{key}: {metric_dict[key]:.4f}, num_scenes: {scene_num}\n"
            logger.info(f"Summary of evaluation results:\n{eval_res}")
            f.write("mean,"+",".join([str(metric_dict[key]) for key in metric_dict.keys()])+"\n")
            f.close()
    torch.distributed.barrier()
    exit(0)

# training
torch.distributed.barrier()
if rank == 0 and config.training.get("perceptual_loss", 0.0) > 0:
    from torchvision.models import vgg19
    vgg = vgg19(weights='DEFAULT') # download vgg model
    del vgg
torch.distributed.barrier()
model.train()
len_dataset = len(dataset) // batch_size_per_gpu * batch_size_per_gpu
cur_epoch = train_steps_done * total_batch_size // len_dataset // grad_accum_steps
datasampler.set_epoch(cur_epoch)
dataloader_iter = iter(dataloader)
param_optim_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
param_optim_list = [p for p in param_optim_dict.values()]
train_steps_start = train_steps_done
param_update_steps_start = train_steps_done // grad_accum_steps
while train_steps_done <= train_steps:
    param_update_steps_done = train_steps_done // grad_accum_steps
    try:
        data = next(dataloader_iter)
    except StopIteration:
        #print("We have exhausted the dataloader iterator, resetting it")
        cur_epoch = param_update_steps_done * total_batch_size // len_dataset
        datasampler.set_epoch(cur_epoch)
        dataloader_iter = iter(dataloader)
        data = next(dataloader_iter)
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(device)
    torch.cuda.synchronize()
    start_time = time.time()
    update_param = ((train_steps_done+1) % grad_accum_steps == 0)
    context = torch.autocast(
        enabled=config.use_amp,
        device_type="cuda",
        dtype=amp_dtype_mapping[config.amp_dtype],
    )
    if update_param:
        with context:
            ret_dict = model(data)
    else:
        with model.no_sync(), context:
            ret_dict = model(data)
    torch.cuda.synchronize()
    fwd_end_time = time.time()
    fwd_time = fwd_end_time - start_time
    fwd_time_str = f"{fwd_time:.2f} s"

    loss_dict = ret_dict['loss']
    total_loss = loss_dict['total_loss']
    scaler.scale(total_loss / grad_accum_steps).backward()
    train_steps_done += 1
    param_update_steps_done = train_steps_done // grad_accum_steps

    skip_optimizer_step = False
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logger.warning(f"NaN or Inf loss detected, skip this iteration")
        skip_optimizer_step = True
        loss_dict['total_loss'] = torch.tensor(0.0).to(device)
    torch.cuda.synchronize()
    bwd_end_time = time.time()
    bwd_time = bwd_end_time - fwd_end_time
    bwd_time_str = f"{bwd_time:.2f} s"

    if update_param and (not skip_optimizer_step):
        # Unscales the gradients of optimizer's assigned parameters in-place
        scaler.unscale_(optimizer)
        with torch.no_grad():
            for n, p in param_optim_dict.items(): 
                if p.grad is None:
                    logger.warning(
                        f"step {train_steps_done} found a None grad for {n}"
                    )
                else:
                    p.grad.nan_to_num_(nan=0.0, posinf=1e-3, neginf=-1e-3)
        total_grad_norm = 0.0
        if config.training.grad_clip_norm > 0:
            grad_clip_norm = config.training.grad_clip_norm
            total_grad_norm = torch.nn.utils.clip_grad_norm_(param_optim_list, max_norm=grad_clip_norm).item()
            if total_grad_norm > grad_clip_norm * 2.0:
                logger.warning(f"step {train_steps_done} grad norm too large {total_grad_norm} > {grad_clip_norm * 2.}")
            allowed_gradnorm = grad_clip_norm * config.training.get("allowed_gradnorm_factor", 5.0)
            if total_grad_norm > allowed_gradnorm:
                skip_optimizer_step = True
                logger.warning(f"step {train_steps_done} grad norm too large {total_grad_norm} > {allowed_gradnorm}, skipping optimizer step")
        if not skip_optimizer_step:
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)        
    torch.cuda.synchronize()
    optim_end_time = time.time()
    optim_time = optim_end_time - bwd_end_time
    optim_time_str = f"{optim_time:.2f} s"

    # logging and checkpointing
    if rank == 0 and update_param:
        gaussian_usage = ret_dict['gaussian_usage'].mean().item()
        loss_dict = {k: v.item() for k, v in loss_dict.items()}
        if param_update_steps_done % config.training.print_every == 0 or param_update_steps_done < param_update_steps_start + 100:
            loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
            loss_str += f", gaussian_usage: {gaussian_usage:.4f}"
            memory_usage = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            logger.info(f"\nStep {param_update_steps_done} / {train_steps // grad_accum_steps}, Epoch {cur_epoch}\n{loss_str}\nfwd_time: {fwd_time_str}, bwd_time: {bwd_time_str}, optim_time: {optim_time_str}, memory: {memory_usage:.2f} MB")
        if param_update_steps_done % config.training.wandb_every == 0 or param_update_steps_done < param_update_steps_start + 100:
            log_dict = {
                "iter": param_update_steps_done,
                "param_update_steps": param_update_steps_done,
                "train_steps": train_steps_done,
                "lr": optimizer.param_groups[0]["lr"],
                "iter_time": time.time() - start_time,
                "grad_norm": total_grad_norm,
                "epoch": cur_epoch,
                "train/gaussian_usage": gaussian_usage,
            }
            log_dict.update({"train/" + k: v for k, v in loss_dict.items()})
            wandb.log(log_dict, step = param_update_steps_done)
        if param_update_steps_done % config.training.checkpoint_every == 0 or train_steps_done == train_steps:
            checkpoint = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_steps_done': train_steps_done,
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, f"checkpoint_{param_update_steps_done:09d}.pt"))
            logger.info(f"Saved checkpoint at step {param_update_steps_done}")
        if param_update_steps_done % config.training.vis_every == 0 or param_update_steps_done < param_update_steps_start + 25:
            save_gaussian = param_update_steps_done % config.training.save_gaussian_every == 0
            save_video = param_update_steps_done % config.training.save_video_every == 0
            model.module.save_visualization(data, ret_dict, os.path.join(checkpoint_dir, f"vis_{param_update_steps_done:09d}"),
                                            save_gaussian = save_gaussian, save_video = save_video)
    torch.distributed.barrier()
torch.distributed.barrier()
destroy_process_group()