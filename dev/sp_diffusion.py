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

def inference(model,scheduler,state,sp_model,tgt_sims,use_sp_guidance,rescale):

    eta = 1.
    info = edict()
    fields = ["sims_delta"]
    for k in fields: info[k] = []

    for t in tqdm(scheduler.timesteps):
        with torch.no_grad():

            # -- compute score --
            score = model(state, t).sample
            # print("noisy_res.abs().mean(): ",noisy_res.abs().mean())

        # -- unpack --
        sched_dict = scheduler.step(score, t, state, eta=eta)
        # next_state = sched_dict.prev_sample
        deno = sched_dict.pred_original_sample

        # -- add guidance --
        sp_scale = get_ddim_scale(scheduler,t,eta)
        if use_sp_guidance:
            sp_grad,sims_delta = superpixel_guidance(deno,sp_model,tgt_sims)
            # next_state = next_state + sp_scale * sp_grad
        else:
            sp_scale = get_ddim_scale(scheduler,t,eta)
            sp_grad,sims_delta = superpixel_guidance(deno,sp_model,tgt_sims)
            sp_grad[...] = 0.
            # sims_delta = th.zeros((len(deno)),device=deno.device).tolist()

        #-- udate --
        score = score + rescale * sp_scale * sp_grad
        sched_dict = scheduler.step(score, t, state, eta=eta)
        state = sched_dict.prev_sample # next_state -> state

        # -- update info --
        info.sims_delta.append(sims_delta)

    return state,info

def get_ddim_scale(scheduler,timestep, eta=0.):

    # 1. get previous step value (=t-1)
    sched = scheduler
    div_tmp = sched.config.num_train_timesteps // sched.num_inference_steps
    prev_timestep = timestep - div_tmp

    # -- alpha prod --
    alpha_prod_t = sched.alphas_cumprod[timestep]
    if prev_timestep >= 0:
        alpha_prod_t_prev = sched.alphas_cumprod[prev_timestep]
    else:
        alpha_prod_t_prev = sched.final_alpha_cumprod

    # -- std dev --
    variance = sched._get_variance(timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)

    # sp_scale = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5)
    sp_scale = math.sqrt(1-alpha_prod_t)
    return sp_scale

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

def superpixel_guidance(deno,sp_model,tgt_sims):

    # -- compute superpixels --
    # _deno = deno.mean(-3,keepdim=True)
    # sims,_,_,ftrs = sp_model(_deno)

    # -- from Eq --
    # def sim_fxn(deno):
    #     deno = (deno + 1)/2.
    #     deno = deno.mean(-3,keepdim=True)
    #     sims,_,_,ftrs = sp_model(deno)
    #     # -- compute grad --
    #     mask = sims!=0
    #     eps = 1e-15
    #     Dkl = th.sum(tgt_sims*th.log(sims+eps)*mask,dim=(-2,-1))
    #     # sp_grad = torch.autograd.grad(Dkl,deno)[0]
    #     return Dkl
    # sp_grad = -torch.autograd.functional.jacobian(sim_fxn, deno)[0]

    # -- v2 [wrong quantity] --
    # deno = deno.requires_grad_(True)
    # deno = (deno + 1)/2.
    # deno = deno.mean(-3,keepdim=True)
    # sims,_,_,ftrs = sp_model(deno)
    # mask = sims!=0
    # eps = 1e-15
    # Dkl = th.sum(tgt_sims*th.log(sims+eps)*mask)
    # sp_grad_v2 = torch.autograd.grad(Dkl,deno)[0]
    # print(sp_grad_v2.abs())

    # -- info --
    # print("delta: ",th.mean((sp_grad-sp_grad_v2).abs()))


    # # print(sp_grad.abs())
    # # print(sp_grad.abs().mean(),sp_grad.abs().std())
    # # print(sp_grad_v2.abs().mean(),sp_grad_v2.abs().std())
    # # exit()
    # rescale = 1
    # return rescale*sp_grad/alpha,sims_delta

    # -- compute grad --
    # print(deno.min(),deno.max())
    deno = deno.requires_grad_(True)
    _deno = (deno + 1)/2.
    _deno = _deno.mean(-3,keepdim=True)
    sims,_,_,ftrs = sp_model(_deno)
    ftrs = ftrs.detach()
    scale = sp_model.affinity_softmax
    # deno = deno.requires_grad_(True)
    dists = get_dists(_deno,ftrs,sp_model.stoken_size[0],sp_model.M)
    # print(dists.abs().mean(),dists.abs().std())
    dists = dists.reshape_as(sims)
    dists = th.sum((tgt_sims - sims) * dists, dim=(-1,-2))
    th.autograd.backward(dists,th.ones_like(dists),inputs=deno)
    Ddists = deno.grad
    # print("Ddists.shape: ",Ddists.shape)
    # delta_sims = tgt_sims - sims
    # print("-"*10)
    # print(tgt_sims[0,15,15,:4,:4])
    # print(sims[0,15,15,:4,:4])
    # print(dists[0,15,15,:4,:4])
    # print("delta_sims.abs().mean(): ",delta_sims.abs().mean())
    # print(dists.abs().mean())

    # -- info --
    mask = sims!=0
    sims_delta = ((tgt_sims - sims)*mask).flatten(1).abs().sum(-1)
    sims_delta = sims_delta / mask.flatten(1).sum(-1)
    sims_delta = sims_delta.tolist()
    # # print(sims_delta)
    # # exit()

    # -- grad --
    # Dkl = th.mean(th.sum((tgt_sims - sims) * Ddists,dim=(-1,-2)))
    # sp_grad = -torch.autograd.grad(Dkl,deno)[0]
    # sp_grad = th.sum((tgt_sims - sims) * Ddists,dim=(-1,-2))

    sp_grad = Ddists
    # scale = 1.
    # rescale = 0.005 #beta/alpha
    # return rescale*(scale*sp_grad/alpha),sims_delta
    # sp_grad = (beta/alpha)*sp_grad
    # print(beta,alpha,sp_grad.abs().mean(),sp_grad.abs().std(),sp_grad.min(),sp_grad.max())
    return sp_grad,sims_delta

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

def load_sp_model():


    # -- init config --
    import cache_io
    tr_fn = "exps/trte_deno/train_ssn_again.cfg"
    te_fn = "exps/trte_deno/test_sp_eval_again.cfg"
    tr_exp = cache_io.fill_test_shell(tr_fn,te_fn)
    read_test = cache_io.read_test_config.run
    _exps = read_test(tr_exp,".cache_io_exps/trte_deno/test_sp",
                    reset=True,skip_dne=True)
    cfg = _exps[-1]

    # -- augment cfg --
    from superpixel_paper.utils import metrics,extract_defaults
    cfg.model_uuid = cfg.tr_uuid
    _cfg = cfg

    # -- load model --
    from superpixel_paper.deno_trte.train import extract_defaults
    from superpixel_paper.spa_config import config_via_spa
    from superpixel_paper.deno_trte.train import load_model as _load_model
    _cfg = extract_defaults(_cfg)
    config_via_spa(_cfg)
    model = _load_model(_cfg)
    print(model)

    # -- load weights --
    _cfg.log_path = "./output/deno/train/"
    ckpt_path = Path(_cfg.log_path) / "checkpoints" / cfg.model_uuid
    chkpt_files = glob.glob(os.path.join(ckpt_path, "*.ckpt"))
    chkpt = torch.load(chkpt_files[-1])
    # N=len("module.")
    state_dict = chkpt['model_state_dict']
    model.load_state_dict(state_dict)
    model = model.cuda()

    return model

def load_sp_model_v0(stride,scale,M):
    from superpixel_paper.models.sp_modules import GenSP
    model = GenSP(n_iter=5,M=M,stoken_size=stride,
                  affinity_softmax=scale,use_grad=True,
                  gen_sp_type="reshape",return_ftrs=True)
    return model

def get_dists(img,ftrs,stride,M):
    # -- imports --
    from superpixel_paper.models.sp_modules import calc_init_centroid
    from superpixel_paper.models.sp_modules import get_abs_indices
    from superpixel_paper.models.sp_modules import PairwiseDistFunction
    from superpixel_paper.utils import append_grid

    # -- set-up --
    H,W = img.shape[-2:]
    sW,sH = W//stride,H//stride
    num_spix = sW*sH
    _, lmap = calc_init_centroid(img, sW, sH)
    amap = get_abs_indices(lmap, sW)
    mask = (amap[1] >= 0) * (amap[1] < num_spix)
    img = append_grid(img[:,None],M/stride)[:,0]

    # -- reshape --
    img = img.reshape(*img.shape[:2], -1)

    # -- compute pwd --
    pwd_fxn = PairwiseDistFunction.apply
    dists = pwd_fxn(img, ftrs, lmap, sW, sH)
    dists = th.where(dists>1e10,0,dists)

    # -- sample only relevant affinity --
    dists = dists.reshape(-1)
    sparse_dists = th.sparse_coo_tensor(amap[:, mask], dists[mask])
    dists = sparse_dists.to_dense().contiguous()

    # -- reshape to match sims --
    shape_str = 'b (sh sw) (h w) -> b h w sh sw'
    dists = rearrange(dists,shape_str,h=H,sh=sH)

    return dists

def get_tgt_sp_celeb(sp_model,B,H,W):
    assert (H==W) and (H==256), "Must be 256."
    root = Path("/home/gauenk/Documents/data/celebhqa_256/images/")
    fns = [fn for fn in root.iterdir() if fn.suffix == ".jpg"]
    idx_list = np.random.permutation(len(fns))[:B]
    imgs,sims = [],[]
    for idx in idx_list:
        img_fn = fns[idx]
        img = Image.open(img_fn).convert("RGB")
        img = th.from_numpy(np.array(img))/255.
        img = rearrange(img,'h w c -> 1 c h w').cuda()

        _img = img.mean(-3,keepdim=True)
        _sims,_,_,_ftrs = sp_model(_img)
        imgs.append(img)
        sims.append(_sims)
    imgs = th.cat(imgs)
    sims = th.cat(sims)
    return imgs, sims

def get_tgt_sp_celeb_woman(sp_model,H,W):
    img_fn = "diff_output/sample/woman.jpg"
    img = Image.open(img_fn).convert("RGB")
    img = th.from_numpy(np.array(img))/255.
    img = rearrange(img,'h w c -> 1 c h w').cuda()

    # from torchvision.transforms import v2
    # transform = v2.RandomCrop(size=(H,W))
    # img = transform(img)
    img = img[...,48:2*H+48:2,76:2*W+76:2]

    _img = img.mean(-3,keepdim=True)
    sims,_,_,ftrs = sp_model(_img)
    return img, sims

def get_tgt_sp_cifar(sp_model,B,H,W):
    root = Path("/home/gauenk/Documents/data/cifar-10/images/")
    fns = [fn for fn in root.iterdir() if fn.suffix == ".png"]
    idx_list = np.random.permutation(len(fns))[:B]
    imgs,sims = [],[]
    for idx in idx_list:
        img = Image.open(fns[idx]).convert("RGB")
        img = th.from_numpy(np.array(img))/255.
        img = rearrange(img,'h w c -> 1 c h w').cuda()
        _img = img.mean(-3,keepdim=True)
        _sims,_,_,ftrs = sp_model(_img)
        imgs.append(img)
        sims.append(_sims)
    imgs = th.cat(imgs)
    sims = th.cat(sims)
    return imgs, sims

def get_tgt_sp(sp_model,H,W):
    root = Path("data/sr/BSD500/images/all/")
    fns = [fn for fn in root.iterdir() if fn.suffix == ".jpg"]
    idx = np.random.permutation(len(fns))[10]
    # print(fns,idx)
    # img_fn = "data/sr/BSD500/images/all/8023.jpg"
    img_fn = str(fns[idx])
    img = Image.open(img_fn).convert("RGB")
    img = th.from_numpy(np.array(img))/255.
    img = rearrange(img,'h w c -> 1 c h w').cuda()

    from torchvision.transforms import v2
    transform = v2.RandomCrop(size=(H,W))
    img = transform(img)

    sims,_,_,ftrs = sp_model(img)
    return img, sims

def get_tgt_sp(ddpm_name,sp_model,B,H,W):
    if "celeb" in ddpm_name:
        return get_tgt_sp_celeb(sp_model,B,H,W)
    elif "cifar10" in ddpm_name:
        return get_tgt_sp_cifar(sp_model,B,H,W)
    else:
        return get_tgt_sp_rand(sp_model,H,W)

def run_exp(exp_name,seed,B,nsteps,use_sp_guidance,
            ddpm_name,stride,scale,M,rescale):

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
    sp_model = load_sp_model_v0(stride,scale,M)
    model = UNet2DModel.from_pretrained(ddpm_name).to("cuda")
    scheduler.set_timesteps(nsteps)

    # -- sample config --
    # B = 20
    sample_size = model.config.sample_size
    noise = torch.randn((B, 3, sample_size, sample_size), device="cuda")

    # -- sample superpixel --
    H,W = sample_size,sample_size
    sp_img,tgt_sims = get_tgt_sp(ddpm_name,sp_model,B,H,W)
    # sp_img, tgt_sims = get_tgt_sp(sp_model,H,W)
    # print(sp_img.shape)
    # exit()
    save_image("sp_img",sp_img,False)
    # exit()
    if tgt_sims.shape[0] == 1:
        tgt_sims = repeat(tgt_sims,'1 h w sh sw -> b h w sh sw',b=B)

    # -- inference --
    sample,info = inference(model,scheduler,noise,sp_model,tgt_sims,
                            use_sp_guidance,rescale)
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
    for i in range(num):
        name_i = name + "_%02d" % i
        run_exp(name_i,seed+i,*args)

def main():

    # -- optional --
    # B = 200
    B = 16
    # nsteps = 5000
    # nsteps = 1000
    nsteps = 100
    seed = 456
    # ddpm_name = "google/ddpm-cat-256"
    ddpm_name = "google/ddpm-celebahq-256"
    # ddpm_name = "google/ddpm-cifar10-32"
    stride = 16
    scale = 2
    M = 1e-8
    rescale = 1.
    N = 1e4//B

    # run_exp("cond_dev",seed,B,nsteps,True,ddpm_name,stride,scale,M,rescale)
    # run_exp("cond_dev2",seed,B,nsteps,True,ddpm_name,stride,scale,M,rescale)
    # run_exp("cond_dev3",seed,B,nsteps,True,ddpm_name,stride,scale,M,rescale)
    # run_exp("cond_dev",seed,B,nsteps,True,ddpm_name,stride,scale,M,rescale)

    print("num: ",N)
    run_batched_exp("cond_dev",N,seed,B,nsteps,True,ddpm_name,stride,scale,M,rescale)
    # run_batched_exp("stand_dev",N,seed,B,nsteps,False,ddpm_name,stride,scale,M,rescale)

    # run_exp("stand_dev",seed,B,nsteps,False,ddpm_name,stride,scale,M,rescale)
    # run_exp("stand_dev2",seed,B,nsteps,False,ddpm_name,stride,scale,M,rescale)
    # run_exp("stand_dev3",seed,B,nsteps,False,ddpm_name,stride,scale,M,rescale)

if __name__ == "__main__":
    main()
