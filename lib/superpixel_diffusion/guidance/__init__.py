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
from superpixel_paper.models.sp_modules import dists_to_sims,get_dists,expand_dists

def superpixel_guidance(state,sp_model,tgt_sims,tgt_ftrs,use_ftrs):
    if use_ftrs:
        return superpixel_guidance_ftrs(state,sp_model,tgt_sims,tgt_ftrs)
    else:
        return superpixel_guidance_slic(state,sp_model,tgt_sims)

def superpixel_guidance_ftrs(state,sp_model,tgt_sims,tgt_ftrs):

    # -- compute current sims --
    H,W = state.shape[-2:]
    state = state.requires_grad_(True)
    # _state = (state + 1)/2.
    scale = sp_model.affinity_softmax
    stride = sp_model.stoken_size[0]
    dists = get_dists(state,tgt_ftrs,stride,sp_model.M)
    sims = dists_to_sims(dists,H,W,scale,stride)
    # dists = expand_dists(dists,state,stride)
    # print(sims[0,0,0,:3,:3])
    # print(sims[0,10,10,:3,:3])
    # print("-"*20)
    # print(dists[0,0,0,:3,:3])
    # print(dists[0,10,10,:3,:3])
    # print("-"*20)
    # print(sims.shape,dists.shape)
    # exit()
    # sims = dists_to_sims(dists,scale,stride)

    # -- compute expectation --
    eps = 1e-15
    expectation = th.sum((tgt_sims * th.log(sims+eps))*(tgt_sims>0),dim=(-2,-1))
    th.autograd.backward(expectation,th.ones_like(expectation),inputs=state)
    Ddists = state.grad
    sp_grad = Ddists

    # -- info --
    mask = sims!=0
    sims_delta = ((tgt_sims - sims)*mask).flatten(1).abs().sum(-1)
    sims_delta = sims_delta / mask.flatten(1).sum(-1)
    sims_delta = sims_delta.tolist()

    return sp_grad,sims_delta

def superpixel_guidance_slic(state,sp_model,tgt_sims):

    # -- compute superpixels --
    # _state = state.mean(-3,keepdim=True)
    # sims,_,_,ftrs = sp_model(_state)

    # -- from Eq --
    # def sim_fxn(state):
    #     _state = (state + 1)/2.
    #     # state = state.mean(-3,keepdim=True)
    #     sims,_,_,ftrs = sp_model(_state)
    #     # -- compute grad --
    #     mask = sims!=0
    #     eps = 1e-15
    #     Dkl = th.sum(tgt_sims*th.log(sims+eps)*mask,dim=(-2,-1))
    #     # sp_grad = torch.autograd.grad(Dkl,state)[0]
    #     return Dkl
    # sp_grad = -torch.autograd.functional.jacobian(sim_fxn, state)[0]
    # sp_grad = -torch.autograd.functional.jacobian(sim_fxn, state)[0]


    # -- v2 [wrong quantity] --
    # state = state.requires_grad_(True)
    # state = (state + 1)/2.
    # state = state.mean(-3,keepdim=True)
    # sims,_,_,ftrs = sp_model(state)
    # mask = sims!=0
    # eps = 1e-15
    # Dkl = th.sum(tgt_sims*th.log(sims+eps)*mask)
    # sp_grad_v2 = torch.autograd.grad(Dkl,state)[0]
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
    # print(state.min(),state.max())
    state = state.requires_grad_(True)
    _state = (state + 1)/2.
    # _state = _state.mean(-3,keepdim=True)
    sims,_,_,ftrs = sp_model(_state)
    eps = 1e-15
    expectation = th.sum((tgt_sims * th.log(sims+eps))*(tgt_sims>0),dim=(-2,-1))
    th.autograd.backward(expectation,th.ones_like(expectation),inputs=state)
    Ddists = state.grad
    sp_grad = Ddists

    # -- old --
    # ftrs = ftrs.detach()
    # scale = sp_model.affinity_softmax
    # # state = state.requires_grad_(True)
    # dists = get_dists(_state,ftrs,sp_model.stoken_size[0],sp_model.M)
    # # print(dists.abs().mean(),dists.abs().std())
    # dists = dists.reshape_as(sims)
    # dists = th.sum((tgt_sims - sims) * dists, dim=(-1,-2))
    # th.autograd.backward(dists,th.ones_like(dists),inputs=state)
    # Ddists = state.grad

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
    # sp_grad = -torch.autograd.grad(Dkl,state)[0]
    # sp_grad = th.sum((tgt_sims - sims) * Ddists,dim=(-1,-2))

    # sp_grad = Ddists
    # scale = 1.
    # rescale = 0.005 #beta/alpha
    # return rescale*(scale*sp_grad/alpha),sims_delta
    # sp_grad = (beta/alpha)*sp_grad
    # print(beta,alpha,sp_grad.abs().mean(),sp_grad.abs().std(),sp_grad.min(),sp_grad.max())
    return sp_grad,sims_delta

# def compute_sp_target(imgs,sp_model):
#     sims,_,_,ftrs = sp_model(_img)

def load_sp_model(version,stride,scale,M):
    if version == "gensp":
        from superpixel_paper.models.sp_modules import GenSP
        model = GenSP(n_iter=5,M=M,stoken_size=stride,
                      affinity_softmax=scale,use_grad=True,
                      gen_sp_type="reshape",return_ftrs=True)
    else:
        raise KeyError(f"Uknown version [{version}]")
    return model

# def load_sp_model():


#     # -- init config --
#     import cache_io
#     tr_fn = "exps/trte_deno/train_ssn_again.cfg"
#     te_fn = "exps/trte_deno/test_sp_eval_again.cfg"
#     tr_exp = cache_io.fill_test_shell(tr_fn,te_fn)
#     read_test = cache_io.read_test_config.run
#     _exps = read_test(tr_exp,".cache_io_exps/trte_deno/test_sp",
#                     reset=True,skip_dne=True)
#     cfg = _exps[-1]

#     # -- augment cfg --
#     from superpixel_paper.utils import metrics,extract_defaults
#     cfg.model_uuid = cfg.tr_uuid
#     _cfg = cfg

#     # -- load model --
#     from superpixel_paper.deno_trte.train import extract_defaults
#     from superpixel_paper.spa_config import config_via_spa
#     from superpixel_paper.deno_trte.train import load_model as _load_model
#     _cfg = extract_defaults(_cfg)
#     config_via_spa(_cfg)
#     model = _load_model(_cfg)
#     print(model)

#     # -- load weights --
#     _cfg.log_path = "./output/deno/train/"
#     ckpt_path = Path(_cfg.log_path) / "checkpoints" / cfg.model_uuid
#     chkpt_files = glob.glob(os.path.join(ckpt_path, "*.ckpt"))
#     chkpt = torch.load(chkpt_files[-1])
#     # N=len("module.")
#     state_dict = chkpt['model_state_dict']
#     model.load_state_dict(state_dict)
#     model = model.cuda()

#     return model


def get_sp_ddpm_scale_train(scheduler, timesteps, eta=0.):
    device = timesteps.device
    sp_scales = []
    for timestep in timesteps:
        scale = get_sp_ddpm_scale_train_v0(scheduler, int(timestep.item()), eta=0.)
        sp_scales.append(scale)
    sp_scales = th.tensor(sp_scales).to(device)
    return sp_scales

def get_sp_ddpm_scale_train_v0(scheduler, timestep, eta=0.):

    # -- misc --
    sched = scheduler
    prev_tstep = sched.previous_timestep(timestep)

    # -- alpha prod --
    alpha_prod_t = sched.alphas_cumprod[timestep]
    acp,final = sched.alphas_cumprod,sched.one
    if prev_tstep>=0:
        alpha_prod_prev = sched.alphas_cumprod[prev_tstep]
    else:
        alpha_prod_prev = final
    # alpha_prod_prev = th.where(prev_tsteps>=0,sched.alphas_cumprod[prev_tsteps],final)
    # alpha_prod_prev = alpha_prod_prev.to(timesteps.device)
    # if prev_timestep >= 0:
    #     alpha_prod_t_prev = sched.alphas_cumprod[prev_tsteps]
    # else:
    #     alpha_prod_t_prev = sched.final_alpha_cumprod

    # -- std dev --
    # variance = []
    # for i in range(len(timesteps)):
    #     variance_i = sched._get_variance(timesteps[i], prev_tsteps[i])
    #     variance.append(variance_i)
    # variance = th.tensor(variance).to(timesteps.device)
    variance = sched._get_variance(timestep, prev_tstep)
    std_dev_t = eta * variance ** (0.5)

    sp_scale = (1 - alpha_prod_prev - std_dev_t**2) ** (0.5)
    # sp_scale = th.sqrt(1-alpha_prod_t)
    return sp_scale

def get_sp_ddpm_scale_train_v1(scheduler, timesteps, eta=0.):

    # 1. get previous step value (=t-1)
    sched = scheduler
    prev_tsteps = sched.previous_timestep(timesteps)

    # -- alpha prod --
    alpha_prod_t = sched.alphas_cumprod[timesteps]
    acp,final = sched.alphas_cumprod,sched.one
    alpha_prod_prev = th.where(prev_tsteps>=0,sched.alphas_cumprod[prev_tsteps],final)
    alpha_prod_prev = alpha_prod_prev.to(timesteps.device)
    # if prev_timestep >= 0:
    #     alpha_prod_t_prev = sched.alphas_cumprod[prev_tsteps]
    # else:
    #     alpha_prod_t_prev = sched.final_alpha_cumprod

    # -- std dev --
    variance = []
    for i in range(len(timesteps)):
        variance_i = sched._get_variance(timesteps[i], prev_tsteps[i])
        variance.append(variance_i)
    variance = th.tensor(variance).to(timesteps.device)
    std_dev_t = eta * variance ** (0.5)

    sp_scale = (1 - alpha_prod_prev - std_dev_t**2) ** (0.5)
    # sp_scale = th.sqrt(1-alpha_prod_t)
    return sp_scale


def get_sp_ddim_scale_inference(scheduler, timestep, eta=0.):

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
    # sp_scale = math.sqrt(1 - alpha_prod_t)
    sp_scale = math.sqrt(1 - alpha_prod_t - std_dev_t**2)
    return sp_scale

