"""

  Simple Diffusion Example

"""

from diffusers import DDIMScheduler
from diffusers import DDPMScheduler, UNet2DModel
import safetensors
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
import copy
dcopy = copy.deepcopy
from einops import rearrange,repeat
from easydict import EasyDict as edict
import torchvision.utils as tv_utils
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from pathlib import Path
# torch.use_deterministic_algorithms(True)
# from torchvision.utils import make_grid
# torchvision.utils.save_image
from superpixel_paper.models.sp_modules import dists_to_sims,get_dists,expand_dists
from superpixel_diffusion.guidance import superpixel_guidance
from superpixel_diffusion.guidance import get_sp_ddim_scale_inference
from superpixel_diffusion.guidance import load_sp_model

def inference(model,scheduler,state,sp_model,
              tgt_sims,tgt_ftrs,use_sp_guidance,rescale,use_ftrs,
              update_target=False):

    eta = 1.
    info = edict()
    fields = ["sims_delta"]
    for k in fields: info[k] = []
    sched_kwargs = {"eta":eta}
    if isinstance(scheduler,DDPMScheduler):
        sched_kwargs = {}

    for t in tqdm(scheduler.timesteps):
        # with torch.no_grad():

        state = state.clone().requires_grad_(True)
        rescaled_score = model(state, t).sample

        # -- compute score --
        # state = state.requires_grad_(True)
        # print("noisy_res.abs().mean(): ",noisy_res.abs().mean())

        # -- unpack --
        sched_dict = scheduler.step(rescaled_score, t, state, **sched_kwargs)
        # next_state = sched_dict.prev_sample
        deno = sched_dict.pred_original_sample

        # -- add guidance --
        # sp_scale = get_sp_ddim_scale_inference(scheduler,t,eta)
        sp_scale = get_sp_ddpm_scale_inference(scheduler,t,len(deno),deno.device)
        # print(sp_scale)
        # print(state.shape,tgt_sims.shape,tgt_ftrs.shape)
        # print((deno.min().item(),deno.max().item()),
        #       (state.min().item(),state.max().item()),
        #       ((tgt_ftrs).min().item(),(tgt_ftrs).max().item()))
        # exit()
        # sp_input,use_for_grad = deno,state
        sp_input,use_for_grad = state,state
        sp_grad,sims_delta = superpixel_guidance(sp_input,sp_model,
                                                 tgt_sims,tgt_ftrs,use_ftrs,
                                                 use_for_grad=use_for_grad)
        # print("sp_input: ",sp_input.mean().item(),sp_input.min().item(),sp_input.max().item())
        #print("state: ",state.mean().item(),state.min().item(),state.max().item())
        # print("tgt_ftrs: ",tgt_ftrs.mean().item(),tgt_ftrs.min().item(),tgt_ftrs.max().item())
        # print("grad: ",sp_grad.abs().mean().item(),
        #       sp_grad.abs().min().item(),sp_grad.abs().max().item())
        if use_sp_guidance is False: sp_grad[...] = 0.
        # print("grad: ",sp_grad.abs().mean().item(),
        #       sp_grad.abs().min().item(),sp_grad.abs().max().item(),use_sp_guidance)
        # sims_delta = th.ones((len(deno)),device=deno.device).tolist()
        # sp_grad = th.zeros_like(score)

        #-- update --
        # rescale_score is negative of score function.
        # sp_scale = 1.
        alpha,beta,var,s_alpha = get_scales(scheduler,t)
        # print(beta)
        rescaled_score = rescaled_score + rescale * (-beta) * sp_grad
        # rescaled_score = rescaled_score - rescale * sp_grad
        # print(sp_scale)
        sched_dict = scheduler.step(rescaled_score, t, state, **sched_kwargs)
        state = sched_dict.prev_sample.detach() # next_state -> state
        # state = state + rescale * sp_scale * sp_grad

        # -- update target --
        if update_target:
            _deno = deno.detach()
            tgt_sims,_,_,tgt_ftrs = sp_model(_deno)

        # -- update info --
        info.sims_delta.append(sims_delta)

    return state,info

def get_sp_ddpm_scale_inference(scheduler,t,B,device):
    alpha,beta,var,s_alpha = get_scales(scheduler,t)
    return th.tensor([s_alpha*(beta/alpha)]).to(device)[:,None,None,None]

def get_scales(scheduler,t):
    prev_t = scheduler.previous_timestep(t)
    alpha_prod_t = scheduler.alphas_cumprod[t]
    one = scheduler.one
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else one
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    s_alpha = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
    var = scheduler._get_variance(t)
    # (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    return alpha_prod_t ** (0.5),beta_prod_t ** (0.5),var,s_alpha


def save_image(name,input,proc=True,with_grid=True):
    # -- create output dir --
    if not Path(name).exists():
        Path(name).mkdir(parents=True)

    # -- process --
    if proc:
        input = (input / 2. + 0.5).clamp(0,1)

    # -- save all in batch --
    B = input.shape[0]
    for b in range(B):
        image = input[b]
        image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
        image = Image.fromarray(image)
        image.save("%s/%05d.png" % (name,b))

    # -- save a grid --
    if with_grid:
        # print("input.min(),input.max(): ",input.min(),input.max())
        grid = tv_utils.make_grid(input)[None,:]
        tv_utils.save_image(grid,Path(name)/"grid.png")

def get_tgt_sp_celeb(sp_model,B,H,W):
    assert (H==W) and (H==256), "Must be 256."
    root = Path("/home/gauenk/Documents/data/celebhqa_256/images/")
    fns = [fn for fn in root.iterdir() if fn.suffix == ".jpg"]
    idx_list = np.random.permutation(len(fns))[:B]
    imgs = []
    for idx in idx_list:
        img_fn = fns[idx]
        img = Image.open(img_fn)#.convert("RGB")
        img = th.from_numpy(np.array(img))/255.
        img = rearrange(img,'h w c -> 1 c h w').cuda()
        # _img = img.mean(-3,keepdim=True)
        _img = img
        imgs.append(img)
    imgs = th.cat(imgs)
    return imgs

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
    return img

def get_tgt_sp_cifar(sp_model,B,H,W):
    root = Path("/home/gauenk/Documents/data/cifar-10/images/")
    return load_sp_from_root(root,sp_model,B,H,W,ext=".png",flip_colors=True)

def get_tgt_sp_flowers(sp_model,B,H,W):
    root = Path("/home/gauenk/Documents/data/102flowers/images/")
    return load_sp_from_root(root,sp_model,B,H,W,ext=".jpg",resize=True)

def load_sp_from_root(root,sp_model,B,H,W,ext=".png",resize=False,flip_colors=False):
    fns = [fn for fn in root.iterdir() if fn.suffix == ext]
    idx_list = np.random.permutation(len(fns))[:B]
    imgs = []
    for idx in idx_list:
        img = Image.open(fns[idx])
        img = th.from_numpy(np.array(img))/255.
        img = rearrange(img,'h w c -> 1 c h w').cuda()
        if resize:
            img = TF.resize(img,(H,W),InterpolationMode.BILINEAR)
        if flip_colors:
            img = img.flip(-3)
        # _img = img.mean(-3,keepdim=True)
        _img = img
        imgs.append(img)
    imgs = th.cat(imgs)
    return imgs

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
    return img

def get_tgt_sp(ddpm_name,sp_model,B,H,W):
    if "celeb" in ddpm_name:
        imgs = get_tgt_sp_celeb(sp_model,B,H,W)
    elif "cifar10" in ddpm_name:
        imgs = get_tgt_sp_cifar(sp_model,B,H,W)
    elif "flowers" in ddpm_name:
        imgs = get_tgt_sp_flowers(sp_model,B,H,W)
    else:
        imgs = get_tgt_sp_rand(sp_model,H,W)
    imgs = 2.*imgs-1.
    sims,_,_,ftrs = sp_model(imgs)
    return imgs,sims,ftrs

def load_model(model_config_name_or_path,chkpt_path,nsteps,res):
    if model_config_name_or_path is None:
        model = UNet2DModel(
            sample_size=res,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

        chkpt_path = Path(chkpt_path) / ("checkpoint-%d/unet" % nsteps)
        chkpt_path = Path(chkpt_path) / "diffusion_pytorch_model.safetensors"
        chkpt_path = str(chkpt_path)
        # state = th.load(str(chkpt_path))
        # print(list(state.keys()))
        state_dict = safetensors.torch.load_file(chkpt_path)
        model.load_state_dict(state_dict)
        model = model.cuda()
    else:
        # config = UNet2DModel.load_config(model_config_name_or_path)
        model = UNet2DModel.from_pretrained(model_config_name_or_path)
        model = model.cuda()
    return model

def extract_defaults(cfg):
    cfg = dcopy(cfg)
    pairs = {"model_chkpt":None,"update_target":False}
    for key in pairs:
        if key in cfg: continue
        cfg[key] = pairs[key]
    return cfg

def run_exp(cfg):

    # -- init seed --
    # exp_name,seed,B,nsteps,use_sp_guidance,
    #             ddpm_name,stride,scale,M,rescale,use_ftrs,
    #             model_name,model_chkpt

    # -- make base name --
    # cfg.ddpm_sched_name = "google/ddpm-cifar10-32"
    # cfg.stride = 1
    # cfg.scale = 1.
    # cfg.M = 0
    # cfg.rescale = 1.
    # cfg.use_ftrs = True
    # cfg.model_name = None
    # cfg.model_chkpt = "ddpm-ema-flowers-64-sp/checkpoint-2500/unet"
    # cfg.sp_source = "flowers"
    cfg = extract_defaults(cfg)
    ddpm_base = cfg.ddpm_sched_name.split("/")[1]
    sp_source = ddpm_base if cfg.sp_source is None else cfg.sp_source
    base_name = sp_source + "_" + cfg.tag

    # -- set seeds --
    os.environ['PYTHONHASHSEED']=str(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    random.seed(cfg.seed)

    # -- init models --
    # scheduler = DDIMScheduler.from_pretrained(cfg.ddpm_sched_name)
    scheduler = DDPMScheduler.from_pretrained(cfg.ddpm_sched_name)
    sp_model = load_sp_model("gensp",cfg.stride,cfg.scale,cfg.M,cfg.sp_niters)
    # model = UNet2DModel.from_pretrained(ddpm_name).to("cuda")
    # model = load_model(ddpm_name,nsteps,None)
    # model = load_model(None,"ddpm-ema-flowers-64/checkpoint-2500/unet")
    # model = load_model(None,"ddpm-ema-flowers-64-sp/checkpoint-2500/unet")
    model = load_model(cfg.model_name,cfg.model_chkpt,cfg.model_nsteps,64)
    scheduler.set_timesteps(cfg.nsteps)

    # -- sample config --
    # B = 20
    sample_size = model.config.sample_size
    print(sample_size)
    noise = torch.randn((cfg.B, 3, sample_size, sample_size), device="cuda")

    # -- sample superpixel --
    H,W = sample_size,sample_size
    sp_img,tgt_sims,tgt_ftrs = get_tgt_sp(sp_source,sp_model,cfg.B,H,W)
    # print("sp_img.min(),sp_img.max(): ",sp_img.min(),sp_img.max())
    # exit()
    save_image("sp_img",sp_img)
    if tgt_sims.shape[0] == 1:
        tgt_sims = repeat(tgt_sims,'1 h w sh sw -> b h w sh sw',b=B)

    # -- some thinking --
    # state = sp_img
    # # state = state #+ th.randn_like(state)
    # # print(state.min(),state.max())
    # # print(tgt_ftrs.min(),tgt_ftrs.max())
    # grad,_ = superpixel_guidance(state,sp_model,tgt_sims,tgt_ftrs,True)
    # print(grad.abs().mean(),grad.abs().min(),grad.abs().max())
    # exit()

    # -- inference --
    sample,info = inference(model,scheduler,noise,sp_model,
                            tgt_sims,tgt_ftrs,
                            cfg.use_sp_guidance,cfg.rescale,cfg.use_ftrs,
                            cfg.update_target)

    # -- viz delta superpixel sims --
    format_info(info)

    # -- save images --
    exp_name = cfg.exp_name
    save_root = Path("diff_output") / base_name / exp_name
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

def run_batched_exp(cfg,num):
    print(f"Running {cfg.exp_name}")
    for i in range(num):
        cfg_i = dcopy(cfg)
        cfg_i.exp_name = cfg.exp_name + "_%02d" % i
        cfg_i.seed = cfg.seed + i
        run_exp(cfg_i)

def run_batched_celebhq():
    # -- config --
    cfg = edict()
    cfg.B = 16
    cfg.seed = 456
    cfg.nsteps = 100
    cfg.ddpm_name = "google/ddpm-celebahq-256"
    cfg.stride = 8
    cfg.scale = 1.
    cfg.M = 0
    cfg.rescale = 1.
    cfg.use_ftrs = True
    # cfg.stride = 16
    # cfg.scale = 2.
    # cfg.M = 1e-8
    # cfg.rescale = 1.
    # cfg.use_ftrs = False
    cfg.model_name = cfg.ddpm_name
    cfg.model_chkpt = None

    # -- run each --
    N = int((1e4-1)//cfg.B+1)
    print("N: ",N)
    cfg.exp_name = "batched/cond_dev"
    cfg.use_sp_guidance = True
    run_batched_exp(cfg,N)
    cfg.exp_name = "batched/stand_dev"
    cfg.use_sp_guidance = False
    run_batched_exp(cfg,N)

def run_batched_cifar10():

    # -- config --
    cfg = edict()
    cfg.seed = 456
    cfg.B = 500
    cfg.nsteps = 1000
    cfg.use_sp_guidance = True
    cfg.ddpm_sched_name = "google/ddpm-cifar10-32"
    cfg.stride = 8
    cfg.scale = 2.
    cfg.M = 0
    cfg.rescale = 1.
    cfg.use_ftrs = True
    cfg.model_name = None
    cfg.model_nsteps = None
    cfg.sp_source = "cifar10"
    cfg.tag = "v0"
    cfg.model_name = cfg.ddpm_sched_name

    # # -- config --
    # cfg = edict()
    # cfg.seed = 456
    # cfg.B = 400
    # cfg.nsteps = 1000
    # cfg.use_sp_guidance = True
    # cfg.ddpm_name = "google/ddpm-cifar10-32"
    # cfg.stride = 1
    # cfg.scale = 1.
    # cfg.M = 0
    # cfg.rescale = 1.
    # cfg.use_ftrs = False
    # cfg.model_name = cfg.ddpm_name
    # cfg.model_chkpt = None

    # -- run each --
    N = int((1e3-1)//cfg.B+1)
    print("N: ",N)
    cfg.exp_name = "batched_v1/cond_dev"
    cfg.use_sp_guidance = True
    run_batched_exp(cfg,N)
    cfg.exp_name = "batched_v1/stand_dev"
    cfg.use_sp_guidance = False
    run_batched_exp(cfg,N)

def main():

    # -- info --
    print("PID: ",os.getpid())

    # -- run batched ones --
    # run_batched_celebhq()
    # run_batched_cifar10()
    # return

    # -- config --
    cfg = edict()
    cfg.seed = 456
    # cfg.seed = 567
    cfg.B = 16
    cfg.nsteps = 100
    cfg.use_sp_guidance = True
    # cfg.ddpm_sched_name = "google/ddpm-cat-256"
    cfg.ddpm_sched_name = "google/ddpm-celebahq-256"
    # cfg.ddpm_sched_name = "google/ddpm-cifar10-32"
    cfg.stride = 8
    cfg.scale = 1.
    cfg.M = 0
    cfg.rescale = 1.
    cfg.use_ftrs = True
    cfg.update_target = False
    cfg.sp_niters = 5
    cfg.model_name = None
    # nsteps = 25500
    # cfg.model_chkpt = "ddpm-ema-flowers-64-sp-03-24/checkpoint-%d/unet" % nsteps
    # cfg.model_chkpt = "ddpm-ema-flowers-64-sp-03-24"
    # cfg.model_chkpt = "ddpm-ema-flowers-64-sp-st8-sc1-m0-fF"
    cfg.model_nsteps = 12000
    # cfg.model_chkpt = "ddpm-ema-flowers-64"
    # cfg.model_chkpt = "ddpm-ema-flowers-64-sp-v0-st8-sc1-m0-fF"
    # cfg.model_nsteps = 12000
    # cfg.model_chkpt = "ddpm-ema-flowers-64-sp-v0p1-st8-sc1-m0-fT"
    cfg.model_chkpt = "ddpm-ema-flowers-64-sp-v0p1-st8-sc1-m0-fT"
    cfg.model_nsteps = 20500
    cfg.sp_source = "flowers"
    # cfg.sp_source = "cifar10"
    # cfg.sp_source = "celeb"
    cfg.tag = "v0"
    # cfg.model_name = cfg.ddpm_sched_name

    # -- conditional --
    cfg.exp_name = "cond_dev"
    cfg.use_sp_guidance = True
    # cfg.model_chkpt = "ddpm-ema-flowers-64"
    # cfg.tag = "v0"
    # run_exp(cfg)
    # cfg.model_chkpt = "ddpm-ema-flowers-64-sp-03-24"
    run_exp(cfg)

    # -- standard --
    cfg.exp_name = "stand_dev"
    cfg.use_sp_guidance = False
    # cfg.model_chkpt = "ddpm-ema-flowers-64"
    # cfg.tag = "v0"
    # run_exp(cfg)
    # cfg.model_chkpt = "ddpm-ema-flowers-64-sp-03-24"
    # cfg.model_chkpt = "ddpm-ema-flowers-64-sp/checkpoint-%d/unet" % nsteps
    run_exp(cfg)

if __name__ == "__main__":
    main()
