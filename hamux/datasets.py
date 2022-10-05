"""
Example Usage:

```
from datasets import DataloadingArgs, DataConfigMNIST

args = DataloadingArgs(
    dataset="torch/MNIST",
    aa=None,
    reprob=0.0,
    vflip=0.,
    hflip=0.,
    scale=(0.8,1.),
    batch_size=10,
    color_jitter=0.,
    validation_batch_size=10_000,
)
data_config = DataConfigMNIST(input_size=(1,28,28))
loader_train, loader_eval = create_dataloaders(args, data_config)
```
"""

import timm
import torch
import numpy as np
from einops import rearrange
from timm.data import create_dataset, create_loader
from dataclasses import dataclass
from pathlib import Path
from typing import *

class ShowImgMixin():
    def show(self, x):
        x = rearrange(x, "... c h w -> ... h w c")
        x = x * torch.tensor(self.std) + torch.tensor(self.mean)
        return np.array(x)
    
@dataclass
class DataConfigMNIST(ShowImgMixin):
    """
    Mean/std from https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data
    """
    input_size:Tuple[int,int,int]=(1,28,28)
    mean:Tuple[float,float,float]=0.
    std:Tuple[float,float,float]=1.
    interpolation:str='bicubic'
    crop_pct:float=0.875
    n_classes:int=10
    
@dataclass
class DataConfigCIFAR10(ShowImgMixin):
    """
    Mean/std from https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data
    """
    input_size:Tuple[int,int,int]=(3,32,32)
    mean:Tuple[float,float,float]=(0.49139968, 0.48215827 ,0.44653124)
    std:Tuple[float,float,float]=(0.24703233, 0.24348505, 0.26158768)
    interpolation:str='bicubic'
    crop_pct:float=0.875
    n_classes:int=10
    
@dataclass
class DataConfigImageNet(ShowImgMixin):
    """
    """
    input_size:Tuple[int,int,int]=(3,224,224)
    mean:Tuple[float,float,float]=(0.485, 0.456, 0.406)
    std:Tuple[float,float,float]=(0.229, 0.224, 0.225)
    interpolation:str='bicubic'
    crop_pct:float=0.875
    n_classes:int=100

@dataclass
class DataloadingArgs:
    data_dir:str=str(Path.home() / "datasets/timm-datasets")
    dataset:str=""
    class_map:str=""
    train_split:str=""
    dataset_download:bool=True
    val_split:str="validation"
    train_split:str="train"
    batch_size:int=32
    epoch_repeats:int=0
    prefetcher:bool=False
    no_aug:bool=False
    reprob:float=0.
    remode:str="pixel"
    recount:int=1
    resplit:bool=False
    scale:Tuple[float,float]=(0.2, 1.0) # Random resize aspect ratio
    ratio:Tuple[float,float]=(3./4., 4./3.)
    hflip:float=0.5
    vflip:float=0.5
    color_jitter:float=0.4
    aa:Optional[str]=None
    # aa="rand"
    aug_repeats:int=0
    workers:int=4
    distributed:bool=False
    pin_mem:bool=False
    use_multi_epochs_loader:bool=False
    worker_seeding:str='all'
    validation_batch_size:Optional[int]=None
        
def create_dataloaders(args, data_config):
    vbatch_size = args.validation_batch_size or args.batch_size
    dataset_train = create_dataset(
        args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        repeats=args.epoch_repeats)

    dataset_eval = create_dataset(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=vbatch_size)
    
    loader_train = create_loader(
        dataset_train,
        input_size=data_config.input_size,
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        # num_aug_splits=num_aug_splits,
        # interpolation=train_interpolation,
        mean=data_config.mean,
        std=data_config.std,
        num_workers=args.workers,
        distributed=args.distributed,
        # collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config.input_size,
        batch_size=vbatch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config.interpolation,
        mean=data_config.mean,
        std=data_config.std,
        num_workers=1,
        distributed=args.distributed,
        crop_pct=data_config.crop_pct,
        pin_memory=args.pin_mem,
    )
    
    return loader_train, loader_eval


""" Example initializations of datasets:

```
## CIFAR10
args = DataloadingArgs(
    dataset="torch/CIFAR10",
    aa=None,
    reprob=0.2,
    vflip=0.2,
    hflip=0.5,
    batch_size=128,
    validation_batch_size=10_000, # Get the entire validation set at once
)
data_config = DataConfigCIFAR10()


## ImageNet
args = DataloadingArgs(
    data_dir=Path.home()/"datasets/timm-datasets/ImageNet100",
    aa=None,
    reprob=0.1,
    vflip=0.0,
    hflip=0.5,
    batch_size=256,
    validation_batch_size=500
)
data_config = DataConfigImageNet(input_size=(3,128,128)) # Feel free to change the input size of our dataset!


## MNIST
args = DataloadingArgs(
    dataset="torch/MNIST",
    aa=None,
    reprob=0.1,
    vflip=0.,
    hflip=0.,
    scale=(0.7,1.),
    batch_size=100,
    # batch_size=2000,
    color_jitter=0.4,
    validation_batch_size=1000,
)
data_config = DataConfigMNIST(input_size=(1,28,28))
```
"""