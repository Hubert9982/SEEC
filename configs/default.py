# from model.distribution import *
import torch.optim as optim
from functools import partial
from utils.schedulers import WarmupScheduler
from datasets.transform import TrainTransform, EvalTransform

from datasets.dataset import ImgMaskDataset

# Training parameters
seed = 1
output_dir = "experiments"
device = "cuda"
batch_size = 64
num_workers = 16
lr = 0.0001
aux_lr = 0.001
# blr = 0.0001
# aux_blr = 0.001
num_epochs = 600
lr_reduce_patience = 30
lr_reduce_factor = 0.9
multistep = False
milestones = [350, 390, 430, 470, 510, 550, 590]
gamma = 0.9
clip_grad = 1.0
p_hflip = 0.5
p_vflip = 0.5
train_path = "./data/DIV2K_train_p128"
val_path = "./data/DIV2K_valid_p128"
prefetch_factor = 8
patch_sz = 64
warmup = False
warmup_epochs = 5

# dataset
transform_train = TrainTransform(patch_sz, p_hflip, p_vflip)
transform_val = EvalTransform(patch_sz)
try:
    train_dataset = ImgMaskDataset(train_path, transform=transform_train)
    val_dataset = ImgMaskDataset(val_path, transform=transform_val)
except Exception as e:
    print("Cannot load dataset")


# Dist Params
dist_backend = "nccl"
dist_url = "env://"
world_size = 1
