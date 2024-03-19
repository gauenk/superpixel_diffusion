"""

  Simple Diffusion Example

"""

from diffusers import DDPMScheduler, UNet2DModel
import torch
import numpy as np
import torch as th
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path
import glob,os
from einops import rearrange,repeat
from easydict import EasyDict as edict

def inference(model,scheduler,state,sp_model,tgt_sims,use_sp_guidance):

    info = edict()
    fields = ["sims_delta"]
    for k in fields: info[k] = []

    for t in tqdm(scheduler.timesteps):
        with torch.no_grad():

            # -- compute score --
            noisy_res = model(state, t).sample
            # print("noisy_res.abs().mean(): ",noisy_res.abs().mean())

        # -- unpack --
        sched_dict = scheduler.step(noisy_res, t, state)
        next_state = sched_dict.prev_sample
        deno = sched_dict.pred_original_sample

        # -- add guidance --
        alpha,beta = get_scales(scheduler,t)
        sp_grad,sims_delta = superpixel_guidance(deno,sp_model,tgt_sims,alpha,beta)
        if use_sp_guidance:
            # alpha,beta = get_scales(scheduler,t)
            # sp_grad,sims_delta = superpixel_guidance(deno,sp_model,tgt_sims,alpha,beta)
            next_state = next_state + sp_grad
        # else:
        #     alpha,beta = get_scales(scheduler,t)
        #     sp_grad,sims_delta = superpixel_guidance(deno,sp_model,tgt_sims,alpha,beta)

        # -- update --
        state = next_state

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

def superpixel_guidance(deno,sp_model,tgt_sims,alpha,beta):
    deno = (deno + 1)/2.
    sims,_,_,ftrs = sp_model(deno)
    scale = sp_model.affinity_softmax
    deno = deno.requires_grad_(True)
    dists = get_dists(deno,ftrs,sp_model.stoken_size[0],sp_model.M)
    dists = dists.reshape_as(sims)
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
    # print(sims_delta)
    # exit()

    # -- grad --
    Dkl = th.mean(th.sum((tgt_sims - sims) * dists,dim=(-1,-2)))
    sp_grad = -torch.autograd.grad(Dkl,deno)[0]
    return 30*(scale*sp_grad)/alpha,sims_delta

def save_image(name,input,proc=True):
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

def load_sp_model_v0():
    from superpixel_paper.models.sp_modules import GenSP
    model = GenSP(n_iter=5,M=1e-5,stoken_size=12,affinity_softmax=10.,
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

def get_tgt_sp_celeb(sp_model,H,W):
    img_fn = "diff_output/sample/woman.jpg"
    img = Image.open(img_fn).convert("RGB")
    img = th.from_numpy(np.array(img))/255.
    img = rearrange(img,'h w c -> 1 c h w').cuda()

    # from torchvision.transforms import v2
    # transform = v2.RandomCrop(size=(H,W))
    # img = transform(img)
    img = img[...,48:2*H+48:2,76:2*W+76:2]

    sims,_,_,ftrs = sp_model(img)
    return img, sims


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

def run_exp(B,nsteps,use_sp_guidance,seed,ddpm_name):

    # -- init seed --
    np.random.seed(seed)
    torch.manual_seed(seed)

    # -- init models --
    scheduler = DDPMScheduler.from_pretrained(ddpm_name)
    sp_model = load_sp_model_v0()
    model = UNet2DModel.from_pretrained(ddpm_name).to("cuda")
    scheduler.set_timesteps(nsteps)

    # -- sample config --
    # B = 20
    sample_size = model.config.sample_size
    noise = torch.randn((B, 3, sample_size, sample_size), device="cuda")

    # -- sample superpixel --
    H,W = sample_size,sample_size
    sp_img,tgt_sims = get_tgt_sp_celeb(sp_model,H,W)
    # sp_img, tgt_sims = get_tgt_sp(sp_model,H,W)
    save_image("sp_img",sp_img,False)
    # exit()
    tgt_sims = repeat(tgt_sims,'1 h w sh sw -> b h w sh sw',b=B)

    # -- inference --
    sample,info = inference(model,scheduler,noise,sp_model,tgt_sims,use_sp_guidance)
    format_info(info)
    save_root = Path("diff_output") / ddpm_name.split("/")[-1]
    if use_sp_guidance:
        save_image(save_root/"cond",sample)
    else:
        save_image(save_root/"standard",sample)

def format_info(info):
    np.set_printoptions(precision=3)
    # -- format sims delta --
    nsteps = len(info.sims_delta)
    delta = np.array([info.sims_delta[t] for t in [0,nsteps-1]])
    delta = np.exp(-delta)
    print(delta)

def main():

    # -- optional --
    B = 20
    nsteps = 50
    seed = 123
    # ddpm_name = "google/ddpm-cat-256"
    ddpm_name = "google/ddpm-celebahq-256"
    run_exp(B,nsteps,True,seed,ddpm_name)
    # run_exp(B,nsteps,False,seed,ddpm_name)
    # run_exp(seed,False)
    # for sp_guidance in [True,False]:
    #     run_exp(sp_guidance)

if __name__ == "__main__":
    main()
