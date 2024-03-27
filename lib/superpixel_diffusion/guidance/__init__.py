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

def superpixel_guidance(state,sp_model,tgt_sims,tgt_ftrs,use_ftrs,
                        use_for_grad=None,use_closed_form=False):
    if use_ftrs is True:
        return superpixel_guidance_ftrs(state,sp_model,tgt_sims,tgt_ftrs,
                                        use_for_grad,use_closed_form)
    else:
        return superpixel_guidance_slic(state,sp_model,tgt_sims,
                                        use_for_grad,use_closed_form)

def superpixel_guidance_ftrs(state,sp_model,tgt_sims,tgt_ftrs,
                             use_for_grad=None,use_closed_form=False):

    # -- compute current sims --
    if use_for_grad is None:
        state = state.clone().requires_grad_(True)
        use_for_grad = state


    # -- compute expectation --
    if use_closed_form:
        sp_grad = compute_min_expectation(sp_model,state,tgt_sims,tgt_ftrs)
    else:
        sp_grad = compute_grad_expectation(sp_model,state,tgt_sims,tgt_ftrs,use_for_grad)

    # -- info --
    mask = sims!=0
    sims_delta = ((tgt_sims - sims)*mask).flatten(1).abs().sum(-1)
    sims_delta = sims_delta / mask.flatten(1).sum(-1)
    sims_delta = sims_delta.tolist()

    return sp_grad,sims_delta

def superpixel_guidance_slic(state,sp_model,tgt_sims,
                             use_for_grad=None,use_closed_form=False):

    # -- compute grad --
    # if use_for_grad is None:
    #     state = state.clone().requires_grad_(True)
    #     use_for_grad = state
    # _state = (state + 1)/2.
    # _state = _state.mean(-3,keepdim=True)
    # sims,_,_,ftrs = sp_model(_state)
    _,_,_,ftrs = sp_model(state)
    ftrs = ftrs.detach()
    return superpixel_guidance_ftrs(state,sp_model,tgt_sims,ftrs,
                                    use_for_grad,use_closed_form)

def compute_min_expectation(sp_model,state,tgt_sims,tgt_ftrs):
    # -- compute sims --
    H,W = state.shape[-2:]
    scale = sp_model.affinity_softmax
    stride = sp_model.stoken_size[0]
    dists = get_dists(state,tgt_ftrs,stride,sp_model.M)
    sims = dists_to_sims(dists,H,W,scale,stride)
    th.autograd.backward(sims,th.ones_like(sims),inputs=dists)
    sim_mat = dists.grad/sims
    th.autograd.backward(dists,th.ones_like(dists),inputs=state)
    sgrad = state.grad + state # \sum N(z;\mu_i,\alpha_i)\mu_i

    return sgrad

def compute_grad_expectation(sp_model,state,tgt_sims,tgt_ftrs,use_for_grad):

    # -- compute sims --
    H,W = state.shape[-2:]
    scale = sp_model.affinity_softmax
    stride = sp_model.stoken_size[0]
    dists = get_dists(state,tgt_ftrs,stride,sp_model.M)
    sims = dists_to_sims(dists,H,W,scale,stride)

    # -- compute expectation --
    eps = 1e-15
    expectation = th.sum((tgt_sims * th.log(sims+eps)),dim=(-2,-1))
    th.autograd.backward(expectation,th.ones_like(expectation),inputs=use_for_grad)
    sp_grad = use_for_grad.grad
    return sp_grad


#     scale = sp_model.affinity_softmax
#     stride = sp_model.stoken_size[0]
#     H,W = state.shape[-2:]
#     dists = get_dists(state,ftrs,stride,sp_model.M)
#     sims = dists_to_sims(dists,H,W,scale,stride)

#     eps = 1e-15
#     expectation = th.sum((tgt_sims * th.log(sims+eps)),dim=(-2,-1))
#     th.autograd.backward(expectation,th.ones_like(expectation),inputs=use_for_grad)
#     sp_grad = use_for_grad.grad
#     # th.autograd.backward(expectation,th.ones_like(expectation),inputs=state)
#     # sp_grad = state.grad

#     # -- info --
#     mask = sims!=0
#     sims_delta = ((tgt_sims - sims)*mask).flatten(1).abs().sum(-1)
#     sims_delta = sims_delta / mask.flatten(1).sum(-1)
#     sims_delta = sims_delta.tolist()

#     return sp_grad,sims_delta

# # def compute_sp_target(imgs,sp_model):
# #     sims,_,_,ftrs = sp_model(_img)

def load_sp_model(version,stride,scale,M,n_iters=5):
    if version == "gensp":
        from superpixel_paper.models.sp_modules import GenSP
        model = GenSP(n_iter=n_iters,M=M,stoken_size=stride,
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

