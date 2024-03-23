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
from superpixel_paper.models.sp_modules import dists_to_sims,get_dists,expand_dists
from superpixel_diffusion.guidance import superpixel_guidance
from superpixel_diffusion.guidance import get_sp_ddim_scale_inference
from superpixel_diffusion.guidance import load_sp_model

def inference(model,scheduler,state,sp_model,
              tgt_sims,tgt_ftrs,use_sp_guidance,rescale,use_ftrs):

    eta = 1.
    info = edict()
    fields = ["sims_delta"]
    for k in fields: info[k] = []

    for t in tqdm(scheduler.timesteps):
        with torch.no_grad():
            rescaled_score = model(state, t).sample

        # -- compute score --
        # state = state.requires_grad_(True)

        # print("noisy_res.abs().mean(): ",noisy_res.abs().mean())

        # -- unpack --
        sched_dict = scheduler.step(rescaled_score, t, state, eta=eta)
        # next_state = sched_dict.prev_sample
        deno = sched_dict.pred_original_sample

        # -- add guidance --
        sp_scale = get_sp_ddim_scale_inference(scheduler,t,eta)
        if use_sp_guidance:
            sp_grad,sims_delta = superpixel_guidance(state,sp_model,
                                                     tgt_sims,tgt_ftrs,use_ftrs)
        else:
            sp_grad,sims_delta = superpixel_guidance(state,sp_model,
                                                     tgt_sims,tgt_ftrs,use_ftrs)
            sp_grad[...] = 0.
            # sims_delta = th.ones((len(deno)),device=deno.device).tolist()
            # sp_grad = th.zeros_like(score)

        #-- udate --
        rescaled_score = rescaled_score + rescale * sp_scale * sp_grad
        sched_dict = scheduler.step(rescaled_score, t, state, eta=eta)
        state = sched_dict.prev_sample # next_state -> state

        # -- update info --
        info.sims_delta.append(sims_delta)

    return state,info


def get_scales(scheduler,t):
    prev_t = scheduler.previous_timestep(t)
    alpha_prod_t = scheduler.alphas_cumprod[t]
    one = scheduler.one
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else one
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t
    # (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    return alpha_prod_t ** (0.5),beta_prod_t ** (0.5)


def save_image(name,input,proc=True,with_grid=True):
    # -- create output dir --
    if not Path(name).exists():
        Path(name).mkdir(parents=True)

    # -- save all in batch --
    B = input.shape[0]
    for b in range(B):
        image = input[b]
        if proc:
            image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
        image = Image.fromarray(image)
        image.save("%s/%05d.png" % (name,b))

    # -- save a grid --
    if with_grid:
        grid = tv_utils.make_grid(input)[None,:]
        tv_utils.save_image(grid,Path(name)/"grid.png")

def get_tgt_sp_celeb(sp_model,B,H,W):
    assert (H==W) and (H==256), "Must be 256."
    root = Path("/home/gauenk/Documents/data/celebhqa_256/images/")
    fns = [fn for fn in root.iterdir() if fn.suffix == ".jpg"]
    idx_list = np.random.permutation(len(fns))[:B]
    imgs,sims,ftrs = [],[],[]
    for idx in idx_list:
        img_fn = fns[idx]
        img = Image.open(img_fn)#.convert("RGB")
        img = th.from_numpy(np.array(img))/255.
        img = rearrange(img,'h w c -> 1 c h w').cuda()

        # _img = img.mean(-3,keepdim=True)
        _img = img
        _sims,_,_,_ftrs = sp_model(_img)
        imgs.append(img)
        sims.append(_sims)
        ftrs.append(_ftrs)
    imgs = th.cat(imgs)
    sims = th.cat(sims)
    ftrs = th.cat(ftrs)
    return imgs, sims, ftrs

def get_tgt_sp_celeb_woman(sp_model,H,W):
    img_fn = "diff_output/sample/woman.jpg"
    img = Image.open(img_fn).convert("RGB")
    img = th.from_numpy(np.array(img))/255.
    img = rearrange(img,'h w c -> 1 c h w').cuda()

    # from torchvision.transforms import v2
    # transform = v2.RandomCrop(size=(H,W))
    # img = transform(img)
    img = img[...,48:2*H+48:2,76:2*W+76:2]

    # _img = img.mean(-3,keepdim=True)
    _img = img
    sims,_,_,ftrs = sp_model(_img)
    return img, sims

def get_tgt_sp_cifar(sp_model,B,H,W):
    root = Path("/home/gauenk/Documents/data/cifar-10/images/")
    fns = [fn for fn in root.iterdir() if fn.suffix == ".png"]
    idx_list = np.random.permutation(len(fns))[:B]
    imgs,sims,ftrs = [],[],[]
    for idx in idx_list:
        img = Image.open(fns[idx])
        img = th.from_numpy(np.array(img))/255.
        img = rearrange(img,'h w c -> 1 c h w').cuda()
        img = img.flip(-3)
        # _img = img.mean(-3,keepdim=True)
        _img = img
        _sims,_,_,_ftrs = sp_model(_img)
        imgs.append(img)
        sims.append(_sims)
        ftrs.append(_ftrs)
    imgs = th.cat(imgs)
    sims = th.cat(sims)
    ftrs = th.cat(ftrs)
    return imgs, sims, ftrs

def get_tgt_sp(sp_model,H,W):
    root = Path("data/sr/BSD500/images/all/")
    fns = [fn for fn in root.iterdir() if fn.suffix == ".jpg"]
    idx = np.random.permutation(len(fns))[10]
    # print(fns,idx)
    # img_fn = "data/sr/BSD500/images/all/8023.jpg"
    img_fn = str(fns[idx])
    img = Image.open(img_fn)
    img = th.from_numpy(np.array(img))/255.
    img = rearrange(img,'h w c -> 1 c h w').cuda()

    from torchvision.transforms import v2
    transform = v2.RandomCrop(size=(H,W))
    img = transform(img)

    sims,_,_,ftrs = sp_model(img)
    return img, sims, ftrs

def get_tgt_sp(ddpm_name,sp_model,B,H,W):
    if "celeb" in ddpm_name:
        return get_tgt_sp_celeb(sp_model,B,H,W)
    elif "cifar10" in ddpm_name:
        return get_tgt_sp_cifar(sp_model,B,H,W)
    else:
        return get_tgt_sp_rand(sp_model,H,W)

def run_exp(exp_name,seed,B,nsteps,use_sp_guidance,
            ddpm_name,stride,scale,M,rescale,use_ftrs):

    # -- init seed --
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # -- init models --
    scheduler = DDIMScheduler.from_pretrained(ddpm_name)
    # scheduler = DDPMScheduler.from_pretrained(ddpm_name)
    sp_model = load_sp_model("gensp",stride,scale,M)
    model = UNet2DModel.from_pretrained(ddpm_name).to("cuda")
    scheduler.set_timesteps(nsteps)

    # -- sample config --
    # B = 20
    sample_size = model.config.sample_size
    noise = torch.randn((B, 3, sample_size, sample_size), device="cuda")

    # -- sample superpixel --
    H,W = sample_size,sample_size
    sp_img,tgt_sims,tgt_ftrs = get_tgt_sp(ddpm_name,sp_model,B,H,W)
    # print(tgt_sims.shape,tgt_ftrs.shape)
    # exit()
    # sp_img, tgt_sims = get_tgt_sp(sp_model,H,W)
    # print(sp_img.shape)
    # exit()
    save_image("sp_img",sp_img,False)
    # exit()
    if tgt_sims.shape[0] == 1:
        tgt_sims = repeat(tgt_sims,'1 h w sh sw -> b h w sh sw',b=B)
    # print("tgt_sims.shape: ",tgt_sims.shape)
    # print("tgt_ftrs.shape: ",tgt_ftrs.shape)
    # exit()

    # -- inference --
    sample,info = inference(model,scheduler,noise,sp_model,
                            tgt_sims,tgt_ftrs,
                            use_sp_guidance,rescale,use_ftrs)
    format_info(info)
    save_root = Path("diff_output") / ddpm_name.split("/")[-1] / exp_name
    save_root_i = save_root / "images"
    if not save_root_i.exists(): save_root_i.mkdir(parents=True)
    print(save_root_i)
    save_image(save_root_i,sample,with_grid=False)

    # -- save grid --
    save_root_g = save_root/"grid"
    if not save_root_g.exists(): save_root_g.mkdir(parents=True)
    grid = (tv_utils.make_grid(sample)[None,:] + 1)/2.
    tv_utils.save_image(grid,save_root_g/"grid.png")

def format_info(info):
    np.set_printoptions(precision=3)
    # -- format sims delta --
    nsteps = len(info.sims_delta)
    delta = np.array([info.sims_delta[t] for t in [0,nsteps-1]])
    delta = np.exp(-10*delta)
    print(delta)

def run_batched_exp(name,num,seed,*args):
    print(f"Running {name}")
    for i in range(num):
        name_i = name + "_%02d" % i
        run_exp(name_i,seed+i,*args)

def run_batched_celebhq():
    # -- config --
    B = 16
    nsteps = 100
    seed = 456
    ddpm_name = "google/ddpm-celebahq-256"
    stride = 16
    scale = 2
    M = 1e-8
    rescale = 1.
    N = int((1e4-1)//B+1)
    use_ftrs = False

    # -- run each --
    run_batched_exp("batched/cond_dev",N,seed,B,nsteps,True,
                    ddpm_name,stride,scale,M,rescale,use_ftrs)
    run_batched_exp("batched/stand_dev",N,seed,B,nsteps,False,
                    ddpm_name,stride,scale,M,rescale,use_ftrs)

def run_batched_cifar10():
    # -- config --
    B = 400
    nsteps = 1000
    seed = 456
    ddpm_name = "google/ddpm-cifar10-32"
    stride = 1
    scale = 1
    M = 0#1e-8
    rescale = 1.
    # N = int((1e4-1)//B+1)
    N = int((1e3-1)//B+1)
    use_ftrs = False

    # -- run each --
    print("N: ",N)
    run_batched_exp("batched_v0/cond_dev",N,seed,B,nsteps,True,
                    ddpm_name,stride,scale,M,rescale,use_ftrs)
    run_batched_exp("batched_v0/stand_dev",N,seed,B,nsteps,False,
                    ddpm_name,stride,scale,M,rescale,use_ftrs)

def main():

    # -- info --
    print("PID: ",os.getpid())
    # run_batched_celebhq()
    # run_batched_cifar10()
    # return

    # -- optional --
    # B = 200
    B = 100
    # B = 16
    # nsteps = 5000
    # nsteps = 1000
    nsteps = 100
    seed = 456
    # ddpm_name = "google/ddpm-cat-256"
    # ddpm_name = "google/ddpm-celebahq-256"
    ddpm_name = "google/ddpm-cifar10-32"
    stride = 1
    scale = 1.
    M = 0#1e-10
    rescale = 1.
    use_ftrs = True
    # N = int((1e4-1)//B+1)

    # run_exp("cond_dev",seed,B,nsteps,True,ddpm_name,stride,scale,M,rescale,use_ftrs)
    # run_exp("cond_dev2",seed,B,nsteps,True,ddpm_name,stride,scale,M,rescale,use_ftrs)
    # run_exp("cond_dev3",seed,B,nsteps,True,ddpm_name,stride,scale,M,rescale,use_ftrs)
    run_exp("cond_dev",seed,B,nsteps,True,ddpm_name,stride,scale,M,rescale,use_ftrs)
    # print("num: ",N)
    # run_batched_exp("cond_dev",N,seed,B,nsteps,True,ddpm_name,stride,scale,M,rescale,use_ftrs)
    # run_batched_exp("stand_dev",N,seed,B,nsteps,False,ddpm_name,stride,scale,M,rescale,use_ftrs)

    # run_exp("stand_dev",seed,B,nsteps,False,ddpm_name,stride,scale,M,rescale,use_ftrs)
    # run_exp("stand_dev2",seed,B,nsteps,False,ddpm_name,stride,scale,M,rescale,use_ftrs)
    # run_exp("stand_dev3",seed,B,nsteps,False,ddpm_name,stride,scale,M,rescale,use_ftrs)

if __name__ == "__main__":
    main()
