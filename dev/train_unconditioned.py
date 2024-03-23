"""

  Simple Diffusion Example

"""

from diffusers import DDIMScheduler
from diffusers import DDPMScheduler, UNet2DModel
import os
import torch
import random
import math
import numpy as np
import torch as th
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path
import glob,os
from einops import rearrange,repeat
from easydict import EasyDict as edict
import torchvision.utils as tv_utils
# torch.use_deterministic_algorithms(True)
# from torchvision.utils import make_grid
# torchvision.utils.save_image

def main():

    print("PID: ",os.getpid())

    # run_batched_celebhq()
    # run_batched_cifar10()
    # return


    # -- optional --
    B = 200
    # B = 16
    # nsteps = 5000
    # nsteps = 1000
    nsteps = 300
    seed = 456
    # ddpm_name = "google/ddpm-cat-256"
    # ddpm_name = "google/ddpm-celebahq-256"
    ddpm_name = "google/ddpm-cifar10-32"
    stride = 1
    scale = 1.
    M = 0#1e-10
    rescale = 1.
    # N = int((1e4-1)//B+1)

    # run_exp("cond_dev",seed,B,nsteps,True,ddpm_name,stride,scale,M,rescale)
    # run_exp("cond_dev2",seed,B,nsteps,True,ddpm_name,stride,scale,M,rescale)
    # run_exp("cond_dev3",seed,B,nsteps,True,ddpm_name,stride,scale,M,rescale)
    run_exp("cond_dev",seed,B,nsteps,True,ddpm_name,stride,scale,M,rescale)
    # print("num: ",N)
    # run_batched_exp("cond_dev",N,seed,B,nsteps,True,ddpm_name,stride,scale,M,rescale)
    # run_batched_exp("stand_dev",N,seed,B,nsteps,False,ddpm_name,stride,scale,M,rescale)

    run_exp("stand_dev",seed,B,nsteps,False,ddpm_name,stride,scale,M,rescale)
    # run_exp("stand_dev2",seed,B,nsteps,False,ddpm_name,stride,scale,M,rescale)
    # run_exp("stand_dev3",seed,B,nsteps,False,ddpm_name,stride,scale,M,rescale)

if __name__ == "__main__":
    main()
