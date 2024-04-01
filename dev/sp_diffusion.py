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
import torchvision.io as tv_io
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
              tgt_img,tgt_sims,tgt_ftrs,use_sp_guidance,rescale,use_ftrs,
              update_target=False,use_deno_sp=False,num_sp_steps=5,
              use_l2_guidance=False,use_deno_grad=True):

    eta = 1.
    comp_sims_delta = use_sp_guidance
    if use_sp_guidance is False:
        num_sp_steps = 1
    info = edict()
    fields = ["sims_delta"]
    for k in fields: info[k] = []
    sched_kwargs = {"eta":eta}
    if isinstance(scheduler,DDPMScheduler):
        sched_kwargs = {}

    for t in tqdm(scheduler.timesteps):

        # -- optionally enable grad --
        state = state.requires_grad_(use_deno_grad)
        with torch.set_grad_enabled(use_deno_grad):
            rescaled_score = model(state, t).sample

        # -- unpack --
        sched_dict = scheduler.step(rescaled_score, t, state, **sched_kwargs)
        deno = sched_dict.pred_original_sample
        if use_deno_grad:
            th.autograd.backward(deno,th.ones_like(deno))
            deno_grad = state.grad
        else:
            deno_grad = th.zeros_like(state)
        state = state.requires_grad_(False)
        deno = deno.detach()

        # -- superpixel guidance --
        alpha,beta,var,s_alpha = get_scales(scheduler,t)
        sp_grad,sims_delta = sp_guidance_loop(state,deno,sp_model,
                                              tgt_sims,tgt_ftrs,use_ftrs,
                                              alpha,beta,num_sp_steps,use_deno_sp,
                                              comp_sims_delta)
        if use_sp_guidance is False: sp_grad[...] = 0.

        # -- l2 guidance --
        l2_grad = l2_guidance(state,deno,sp_model,tgt_img,alpha,beta,use_deno_sp)
        if use_l2_guidance is False: l2_grad[...] = 0.

        # -- apply deno grad --
        if use_deno_grad:
            sp_grad = sp_grad * deno_grad
            l2_grad = l2_grad * deno_grad

        #-- update --
        # print(deno_grad.mean(),deno_grad.min(),deno_grad.max())
        rescaled_score = rescaled_score + rescale * (-beta) * sp_grad
        rescaled_score = rescaled_score + rescale * (-beta) * l2_grad
        sched_dict = scheduler.step(rescaled_score, t, state.detach(), **sched_kwargs)
        state = sched_dict.prev_sample.detach() # next_state -> state

        # -- update target --
        if update_target:
            _deno = deno.detach()
            _,_,_,tgt_ftrs,tgt_sims = sp_model(_deno)

        # -- update info --
        info.sims_delta.append(sims_delta)
    print(state.flatten(1).mean(-1))

    return state,info

def sp_guidance_loop(state,deno,sp_model,
                     tgt_sims,tgt_ftrs,use_ftrs,
                     alpha,beta,num_sp_steps,use_deno_sp,comp_sims_delta):
    if comp_sims_delta is False:
        return th.zeros_like(state),th.ones(len(state))
    lr = 1e-2
    eps = 1e-15
    for s in range(num_sp_steps):
        # sp_input = th.clamp(sp_input,-1,1)
        if use_deno_sp:
            sp_input,use_for_grad = deno,state
        else:
            # sp_input = state.requires_grad_(True)/(alpha+eps)
            sp_input = state/(alpha+eps)
            # sp_input = state
            # print(sp_input.mean(),sp_input.max(),sp_input.min())
            sp_input = th.clamp(sp_input,-1,1)
            use_for_grad = sp_input
        sp_grad,sims_delta = superpixel_guidance(sp_input,sp_model,
                                                 tgt_sims,tgt_ftrs,use_ftrs,
                                                 use_for_grad=use_for_grad)
        if num_sp_steps > 1:
            state = (state.detach() + lr*sp_grad)
            use_for_grad = sp_input
    return sp_grad,sims_delta

def l2_guidance(state,deno,sp_model,tgt_img,
                alpha,beta,use_deno_sp):
    lr = 1e-2
    eps = 1e-15
    # sp_input = th.clamp(sp_input,-1,1)
    if use_deno_sp:
        sp_input,use_for_grad = deno,state
    else:
        sp_input = state/(alpha+eps)
        sp_input = th.clamp(sp_input,-1,1)
        use_for_grad = sp_input
    sp_grad = 2*(tgt_img - sp_input)
    return sp_grad

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


def read_images(name,B):
    # -- save all in batch --
    for b in range(B):
        image = Image.fromarray(image)
        if with_images:
            image.save("%s/%05d.png" % (name,b))

def save_image(name,input,proc=True,with_grid=True,with_images=True):
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
        if with_images:
            image.save("%s/%05d.png" % (name,b))

    # -- save a grid --
    if with_grid:
        # print("input.min(),input.max(): ",input.min(),input.max())
        grid = tv_utils.make_grid(input)[None,:]
        tv_utils.save_image(grid,Path(name)/"grid.png")

def get_tgt_sp_celeb_woman(H,W):
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

def get_tgt_sp_celeb(B,H,W):
    assert (H==W) and (H==256), "Must be 256."
    root = name_to_path("celeb")
    # root = Path("/home/gauenk/Documents/data/celebhqa_256/images/")
    return load_sp_from_root(root,B,H,W,flip_colors=False)
    # fns = [fn for fn in root.iterdir() if fn.suffix == ".jpg"]
    # idx_list = np.random.permutation(len(fns))[:B]
    # imgs = []
    # for idx in idx_list:
    #     img_fn = fns[idx]
    #     img = Image.open(img_fn)#.convert("RGB")
    #     img = th.from_numpy(np.array(img))/255.
    #     img = rearrange(img,'h w c -> 1 c h w').cuda()
    #     # _img = img.mean(-3,keepdim=True)
    #     _img = img
    #     imgs.append(img)
    # imgs = th.cat(imgs)
    # return imgs

def get_tgt_sp_cifar(B,H,W):
    root = name_to_path("cifar10")
    return load_sp_from_root(root,B,H,W,flip_colors=True)

def get_tgt_sp_flowers(B,H,W):
    root = name_to_path("flowers")
    return load_sp_from_root(root,B,H,W,resize=True)

def name_to_path(name):
    if "celeb" in name:
        return Path("/home/gauenk/Documents/data/celebhqa_256/images/")
    elif "cifar10" in name:
        return Path("/home/gauenk/Documents/data/cifar-10/images/")
    elif "flowers" in name:
        return Path("/home/gauenk/Documents/data/102flowers/images/")
    else:
        raise KeyError(f"Uknown path name [{name}]")

def load_sp_from_root(root,B,H,W,resize=False,flip_colors=False):
    fns = [fn for fn in root.iterdir() if fn.suffix in [".jpg",".png"]]
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
        # print(img.shape)
        imgs.append(img)
    imgs = th.cat(imgs)
    return imgs

# def get_tgt_sp(sp_model,H,W):
#     root = Path("data/sr/BSD500/images/all/")
#     fns = [fn for fn in root.iterdir() if fn.suffix == ".jpg"]
#     idx = np.random.permutation(len(fns))[10]
#     # print(fns,idx)
#     # img_fn = "data/sr/BSD500/images/all/8023.jpg"
#     img_fn = str(fns[idx])
#     img = Image.open(img_fn)
#     img = th.from_numpy(np.array(img))/255.
#     img = rearrange(img,'h w c -> 1 c h w').cuda()

#     from torchvision.transforms import v2
#     transform = v2.RandomCrop(size=(H,W))
#     img = transform(img)
#     return img

def sample_random_images(ddpm_name,B,H,W):
    if "celeb" in ddpm_name:
        imgs = get_tgt_sp_celeb(B,H,W)
    elif "cifar10" in ddpm_name:
        imgs = get_tgt_sp_cifar(B,H,W)
    elif "flowers" in ddpm_name:
        imgs = get_tgt_sp_flowers(B,H,W)
    else:
        raise ValueError(f"Uknown data name [{ddpm_name}]")
    imgs = 2.*imgs-1.
    return imgs

def get_tgt_sp(ddpm_name,sp_model,B,H,W):
    imgs = sample_random_images(ddpm_name,B,H,W)
    # print("pre: ",imgs.min().item(),imgs.max().item())
    # print(imgs.min().item(),imgs.max().item())
    _,_,_,ftrs,sims = sp_model(imgs)
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
    pairs = {"model_chkpt":None,"update_target":False,"use_deno_sp":False,
             "use_l2_guidance":False,"use_deno_grad":True}
    for key in pairs:
        if key in cfg: continue
        cfg[key] = pairs[key]
    return cfg

def run_exp(cfg,sp_img=None):

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
    if sp_img is None:
        sp_img,tgt_sims,tgt_ftrs = get_tgt_sp(sp_source,sp_model,cfg.B,H,W)
    else:
        _,_,_,tgt_ftrs,tgt_sims = sp_model(sp_img)
    # print(tgt_ftrs.shape)
    # print(tgt_ftrs[0,:,256//8*128//8+128//8])
    # print("sp_img.min(),sp_img.max(): ",sp_img.min(),sp_img.max())
    # exit()
    save_image("sp_img",sp_img,with_images=False)
    if tgt_sims.shape[0] == 1:
        # tgt_sims = repeat(tgt_sims,'1 h w sh sw -> b h w sh sw',b=cfg.B)
        # tgt_sims = repeat(tgt_sims,'1 nine (h w) -> b h w nine',b=cfg.B,h=H,w=W)
        tgt_sims = repeat(tgt_sims,'1 nine hw -> b nine hw',b=cfg.B)
        tgt_ftrs = repeat(tgt_ftrs,'1 f shsw -> b f shsw',b=cfg.B)
    print("tgt_ftrs.shape: ",tgt_ftrs.shape)
    print("tgt_sims.shape: ",tgt_sims.shape)

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
                            sp_img,tgt_sims,tgt_ftrs,
                            cfg.use_sp_guidance,cfg.rescale,cfg.use_ftrs,
                            cfg.update_target,cfg.use_deno_sp,cfg.num_sp_steps,
                            use_l2_guidance=cfg.use_l2_guidance,
                            use_deno_grad=cfg.use_deno_grad)

    # -- viz delta superpixel sims --
    format_info(info)

    # -- save images --
    exp_name = cfg.exp_name
    save_root = Path("diff_output") / base_name / exp_name
    save_root_i = save_root / "images"
    if not save_root_i.exists(): save_root_i.mkdir(parents=True)
    print(save_root_i)
    save_image(save_root_i,sample,with_grid=False)
    # rz_sp_img = save_resized(sp_img,H,W,3)
    # print("delta: ",th.mean((rz_sp_img - sample)**2).item())

    # -- save grid --
    save_root_g = save_root/"grid"
    if not save_root_g.exists(): save_root_g.mkdir(parents=True)
    grid = (tv_utils.make_grid(sample)[None,:] + 1)/2.
    tv_utils.save_image(grid,save_root_g/"grid.png")

    # -- info --
    # print(tgt_ftrs[0,:,256//2*128//2+128//2])
    # print(sp_img[0,:,128,128])
    # print(sample[0,:,128,128])

    return sample

def search_images(stnd,sp_model,use_ftrs,sp_source,
                  max_num=-1,tol=1e-3,max_nochange=1000,stype="lpips"):
    import lpips
    get_lpips = lpips.LPIPS(net='alex').to("cuda")

    def read_fn(fn,flip_colors,device):
        img = Image.open(fn)
        img = th.from_numpy(np.array(img))/255.
        img = rearrange(img,'h w c -> 1 c h w').cuda()
        # if resize:
        #     img = TF.resize(img,(H,W),InterpolationMode.BILINEAR)
        if flip_colors:
            img = img.flip(-3)
        img = 2.*img.to(device)-1
        return img

    # -- compute reference --
    _,_,_,prop_ftrs,prop_sims = sp_model(stnd)

    # -- init --
    device = stnd.device
    root = name_to_path(sp_source)
    print(f"Searching from {str(root)}")
    flip_colors= "cifar10" in sp_source
    fns = [fn for fn in root.iterdir() if fn.suffix in [".jpg",".png"]]
    idx_list = np.random.permutation(len(fns))
    max_num = min(max_num,len(idx_list))
    print(max_num,idx_list)
    if max_num > 0: idx_list = idx_list[:max_num]
    nochange = 0
    index = -1
    min_dists = 1e10*th.ones(len(stnd),device=device)
    min_inds = -1*th.ones(len(stnd),device=device)
    min_dists_prev = min_dists.clone()
    for idx in tqdm(idx_list,desc="searching images."):

        # -- read image --
        img = read_fn(fns[idx],flip_colors,device)

        # -- compute sp --
        # _,_,_,prop_ftrs,prop_sims = sp_model(img)

        # -- compute dists --
        # dists = (stnd - img).flatten(1).pow(2).mean(-1)
        # dists = (prop_sims - tgt_sims).flatten(1).pow(2).mean(-1)
        # if use_ftrs:
        #     dists += (prop_ftrs - tgt_ftrs).flatten(1).pow(2).mean(-1)
        # print(img.min(),img.max(),img.mean())
        # print(stnd.min(),stnd.max(),stnd.mean())
        if stype == "lpips":
            with th.no_grad():
                dists = get_lpips(img,stnd)
            dists = dists.flatten(1).squeeze()
        elif stype == "spmat":
            # with th.no_grad():
            #     dists0 = get_lpips(img,stnd)
            # dists0 = dists0.flatten(1).squeeze()
            # # print(dists0)
            _,_,_,tgt_ftrs,tgt_sims = sp_model(img)
            # print(tgt_sims[:,:,100])
            # print(prop_sims[:,:,100])
            expectation = th.sum(tgt_sims*th.log(prop_sims+1e-15),dim=1)
            dists = -expectation.flatten(1).mean(-1) # [ expect. in [-\infty,0) ]
        else:
            raise ValueError(f"Uknown search type [{stype}]")
        # print(dists)
        # print(min_dists)
        # print(min_inds)
        # print("-"*20)
        # print(dists.shape)
        # print(dists.shape)
        # print(dists)
        # exit()

        # -- compute minimiums --
        min_inds = th.where(dists < min_dists,idx,min_inds)
        min_dists = th.where(dists < min_dists,dists,min_dists)

        # -- stop early if no updates --
        delta_dists = th.abs(min_dists - min_dists_prev).sum()
        if delta_dists < 1e-8: nochange += 1
        else: nochange = 0
        if (max_nochange > 0) and (nochange >= max_nochange): break

        # -- update previous dists --
        min_dists_prev = min_dists.clone()
        # print(min_dists)

        # if (idx % 500) == 0:
        #     print(min_dists)
        #     print(min_inds)

    srch = []
    print(min_dists)
    print(min_inds)
    for idx in min_inds.long().tolist():
        srch.append(read_fn(fns[idx],flip_colors,device))
    srch = th.cat(srch)
    return srch

def format_info(info):
    np.set_printoptions(precision=3)
    # -- format sims delta --
    nsteps = len(info.sims_delta)
    delta = np.array([info.sims_delta[t] for t in [0,nsteps-1]])
    delta = np.exp(-10*delta)
    print(delta)

def save_resized(img,H,W,S):
    img = TF.resize(img,(H//S,W//S),InterpolationMode.BILINEAR)
    img = TF.resize(img,(H,W),InterpolationMode.BILINEAR)
    save_image("resized",img,with_images=True,with_grid=False)
    return img

def run_batched_exp(cfg,num):
    print(f"Running {cfg.exp_name}")
    for i in range(num):
        cfg_i = dcopy(cfg)
        cfg_i.exp_name = cfg.exp_name + "_%02d" % i
        cfg_i.seed = cfg.seed + i
        run_exp(cfg_i)

def read_exp_imgs(cfg):
    # -- unpack --
    cfg = extract_defaults(cfg)
    ddpm_base = cfg.ddpm_sched_name.split("/")[1]
    sp_source = ddpm_base if cfg.sp_source is None else cfg.sp_source
    base_name = sp_source + "_" + cfg.tag

    # -- save images --
    exp_name = cfg.exp_name
    save_root = Path("diff_output") / base_name / exp_name
    save_root_i = save_root / "images"
    if not save_root_i.exists(): save_root_i.mkdir(parents=True)
    print(save_root_i)
    device = "cuda"
    imgs = read_images_from_root(save_root_i,cfg.B,device)
    return imgs

def load_searched_images(root,B,device):
    return read_images_from_root(root,B,device)

def read_images_from_root(root,B,device):
    imgs = []
    for b in range(B):
        img_b = tv_io.read_image(str(Path(root)/("%05d.png"%b)))
        imgs.append(img_b/255.)
    imgs = th.stack(imgs)
    imgs = 2*imgs.to(device)-1
    return imgs

def save_searched_images(root,imgs):
    if not Path(root).exists():
        Path(root).mkdir(parents=True)
    imgs = (imgs+1)/2.
    for b in range(len(imgs)):
        tv_utils.save_image(imgs[b],Path(root)/("%05d.png"%b))

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
    cfg.scale = 1.
    cfg.M = 0
    cfg.rescale = 1.
    cfg.use_ftrs = True
    cfg.model_name = None
    cfg.model_nsteps = None
    cfg.sp_source = "cifar10"
    cfg.tag = "v0"
    cfg.model_name = cfg.ddpm_sched_name
    cfg.sp_niters = 5

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
    N = int((6e4-1)//cfg.B+1)
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
    # cfg.seed = 456
    # cfg.seed = 567
    # cfg.seed = 567+1
    cfg.seed = 567+22
    # cfg.B = 8
    cfg.B = 200
    cfg.nsteps = 100
    cfg.use_sp_guidance = True
    # cfg.ddpm_sched_name = "google/ddpm-cat-256"
    # cfg.ddpm_sched_name = "google/ddpm-celebahq-256"
    cfg.ddpm_sched_name = "google/ddpm-cifar10-32"
    cfg.num_sp_steps = 1
    # cfg.stride = 12
    # cfg.scale = .25

    # cfg.stride = 16
    # cfg.scale = .5
    # cfg.scale = .3
    # cfg.rescale = 1.
    # cfg.use_ftrs = True
    # cfg.use_deno_sp = True

    cfg.M = 0.
    cfg.stride = 16
    # -- works for celeb --
    cfg.use_ftrs = True
    cfg.use_deno_sp = True
    # cfg.scale = .5
    # cfg.scale = .75
    cfg.scale = 1.
    # cfg.rescale = .25
    cfg.rescale = .1
    cfg.sp_niters = 5

    # -- if not using searched images --
    # cfg.scale = 0.5
    cfg.scale = .75

    # -- when use ftrs --
    # cfg.use_ftrs = False
    # cfg.use_deno_sp = True
    # cfg.scale = .75
    # cfg.sp_niters = 5

    # -- a comparison --
    # cfg.M = 50.
    # cfg.stride = 2
    # cfg.use_ftrs = True
    # cfg.use_deno_sp = True
    # cfg.scale = 1.
    # cfg.rescale = .0025
    # cfg.sp_niters = 2

    cfg.update_target = False
    cfg.model_name = None
    # nsteps = 25500
    cfg.model_name = cfg.ddpm_sched_name
    # cfg.model_chkpt = "ddpm-ema-flowers-64-sp-03-24/checkpoint-%d/unet" % nsteps
    # cfg.model_chkpt = "ddpm-ema-flowers-64-sp-03-24"
    # cfg.model_chkpt = "ddpm-ema-flowers-64-sp-st8-sc1-m0-fF"
    cfg.model_nsteps = 12000
    # cfg.model_chkpt = "ddpm-ema-flowers-64"
    # cfg.model_chkpt = "ddpm-ema-flowers-64-sp-v0-st8-sc1-m0-fF"
    # cfg.model_nsteps = 12000
    # cfg.model_chkpt = "ddpm-ema-flowers-64-sp-v0p1-st8-sc1-m0-fT"
    # cfg.model_chkpt = "ddpm-ema-flowers-64-sp-v0p2-st8-sc1-m0-fT"
    # cfg.model_chkpt = "ddpm-ema-flowers-64-sp-v0p2-st8-sc1-m0-fT-dT"
    # cfg.model_nsteps = 20000
    # cfg.sp_source = "flowers"
    # cfg.model_name = None
    cfg.use_deno_grad = False

    cfg.sp_source = "cifar10"
    # cfg.sp_source = "celeb"
    cfg.tag = "v0"

    # -- standard --
    cfg.exp_name = "stand_dev"
    cfg.use_sp_guidance = False
    # cfg.model_chkpt = "ddpm-ema-flowers-64"
    # cfg.tag = "v0"
    # run_exp(cfg)
    # cfg.model_chkpt = "ddpm-ema-flowers-64-sp-03-24"
    # cfg.model_chkpt = "ddpm-ema-flowers-64-sp/checkpoint-%d/unet" % nsteps
    # stnd = run_exp(cfg)
    stnd = read_exp_imgs(cfg)

    # -- searching --
    stype = "lpips"
    stype = "spmat"
    search_root = Path("searched") / stype / str(cfg.seed)

    # sp_scale = cfg.scale
    # sp_stride = 96
    # sp_search_scale = 4.
    # sp_model = load_sp_model("gensp",cfg.stride,sp_search_scale,cfg.M,cfg.sp_niters)
    # srch = search_images(stnd,sp_model,False,cfg.sp_source,max_num=3e5,stype=stype,
    #                      max_nochange=3e3)
    # save_searched_images(search_root,srch)
    # srch = load_searched_images(search_root,cfg.B,"cuda")
    if cfg.sp_source == "cifar10": H,W = 32,32
    else: H,W = 256,256
    srch = sample_random_images(cfg.sp_source,cfg.B,H,W)
    srch = sample_random_images(cfg.sp_source,cfg.B,H,W)
    # srch = None

    # -- conditional --
    cfg.exp_name = "cond_dev"
    cfg.use_sp_guidance = True
    cfg.use_ftrs = True
    # cfg.scale = 0.75
    # cfg.rescale = 1.
    # cfg.use_ftrs = True
    # cfg.scale = 2.
    # cfg.rescale = 0.01

    cfg.use_deno_grad = True
    # cfg.stride = 96
    cfg.stride = 8
    cfg.use_ftrs = True
    # cfg.scale = 2.
    cfg.scale = 4.
    cfg.rescale = 0.1
    # cfg.scale = 1.
    # cfg.scale = .25 # pretty good, but not enough
    # cfg.rescale = 0.05 # too small
    cfg.rescale = 0.5
    # cfg.rescale = 0.0005
    # cfg.rescale = 0.005
    # cfg.rescale = 0.001
    # cfg.rescale = 0.00025
    # cfg.rescale = 0.0001
    # cfg.rescale = 0.00001
    # cfg.rescale = 0.00001
    # cfg.rescale = 0.00005

    # cfg.stride = 96
    # cfg.scale = 5.
    # cfg.use_ftrs = False
    # cfg.rescale = 0.0001

    # cfg.model_chkpt = "ddpm-ema-flowers-64"
    # cfg.tag = "v0"
    # run_exp(cfg)
    # cfg.model_chkpt = "ddpm-ema-flowers-64-sp-03-24"
    # run_exp(cfg,srch)
    imgs_cond = read_exp_imgs(cfg)

    # -- l2 guidance --
    cfg.use_l2_guidance = True
    cfg.use_sp_guidance = False
    cfg.exp_name = "l2_dev"
    # cfg.rescale = .2
    # cfg.rescale = .15
    # cfg.rescale = .05 # looks like a copy
    # cfg.rescale = .01 # still very "copy" like
    cfg.rescale = .005 # pretty good
    # cfg.rescale = .0025 # copy
    # cfg.rescale = .0020 # ???
    # cfg.rescale = .0015 # ???
    # cfg.rescale = .001 # ???
    # cfg.rescale = .005 # looks good
    # cfg.rescale = .0025 # looks good
    # cfg.rescale = .001 # too much
    # cfg.rescale = .0009 # ?
    # cfg.rescale = .0005 # too little
    cfg.use_deno_sp = True
    run_exp(cfg,srch)
    imgs_l2 = read_exp_imgs(cfg)

    # -- compare lpips --
    # print("stnd: ",dists.flatten(1).mean())
    # dists = get_lpips(imgs_cond,srch)
    # print("cond: ",dists.flatten(1).mean())
    # dists = get_lpips(imgs_l2,srch)
    # print("l2: ",dists.flatten(1).mean())

    # -- lpips/fid/is score --
    import lpips
    get_lpips = lpips.LPIPS(net='alex').to("cuda")
    from pytorch_gan_metrics import get_inception_score, get_fid
    if cfg.sp_source == "celeb":
        ref_fn = "sp_data/celebhq_stats.npz"
    elif cfg.sp_source == "cifar10":
        ref_fn = "sp_data/fid_cifar10_stats.npz"
    else:
        raise ValueError("")
    print("Name: [Big] [Info] [Small] [Big] [Small]")

    stnd = (stnd+1)/2.
    IS, IS_std = get_inception_score(stnd,splits=1,use_torch=True) # Inception Score
    FID = get_fid(stnd, ref_fn) # Frechet Inception Distance
    dists = get_lpips(2*stnd-1,srch).mean().item()
    dists_st = get_lpips(2*stnd-1,2*stnd-1).mean().item()
    print("Stnd: ",dists,dists_st,FID,IS,IS_std)

    imgs_cond = (imgs_cond+1)/2.
    IS, IS_std = get_inception_score(imgs_cond,splits=1,use_torch=True)
    FID = get_fid(imgs_cond, ref_fn) # Frechet Inception Distance
    dists = get_lpips(2*imgs_cond-1,srch).mean().item()
    dists_st = get_lpips(2*imgs_cond-1,2*stnd-1).mean().item()
    print("Cond: ",dists,dists_st,FID,IS,IS_std)

    imgs_l2 = (imgs_l2+1)/2.
    IS, IS_std = get_inception_score(imgs_l2,splits=1,use_torch=True) # Inception Score
    FID = get_fid(imgs_l2, ref_fn) # Frechet Inception Distance
    dists = get_lpips(2*imgs_l2-1,srch).mean().item()
    dists_st = get_lpips(2*imgs_l2-1,2*stnd-1).mean().item()
    print("L2: ",dists,dists_st,FID,IS,IS_std)


if __name__ == "__main__":
    main()
