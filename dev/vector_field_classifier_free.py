


import torch as th
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_grid(nsamples,ulim,llim):
    xi,yi = np.mgrid[llim:ulim:nsamples*1j,llim:ulim:nsamples*1j]
    x,y = xi.ravel(),yi.ravel()
    grid = th.from_numpy(np.stack([x,y]).T)
    return grid

def get_gmm_prob(grid,means,cov,pi):
    prob = 0
    for k in range(len(means)):
        prob += pi[k] * get_gaussian_prob(grid,means[k],cov[k])
    return prob

def get_gaussian_prob(grid,mean,cov):
    dim = int(cov.shape[-1])
    inv_cov = th.linalg.pinv(cov)[None,]
    const = np.power(2*np.pi,-dim/2.) * 1./th.sqrt(th.linalg.det(cov))
    delta = grid - mean[None,]
    mat = th.einsum('BNi,Bi ->BN', inv_cov.double(), delta.double())
    probs = th.exp(-1/2 * th.sum(delta * mat,-1))
    return const * probs

def get_implicit_cls_lprob(grid,means,cov,pi,k0):
    # -- compute log prob --
    prob = 0
    for k in range(len(pi)):
        prob += pi[k] * get_gaussian_prob(grid,means[k],cov[k])
    num = pi[k0] * get_gaussian_prob(grid,means[k0],cov[k0])
    eps = 1e-15
    log_prob = th.log(num+eps) - th.log(prob+eps)
    return log_prob

def get_hard_implicit_cls_field(grid,means,cov,pi,k0=0):

    # -- compute log prob --
    grid = grid.clone().requires_grad_(True)
    log_prob = get_implicit_cls_lprob(grid,means,cov,pi,k0)

    # -- backward --
    th.autograd.backward(log_prob,th.ones_like(log_prob))
    return grid.grad

def get_soft_implicit_cls_field(grid,means,cov,pi,probs):

    # -- compute log prob --
    grid = grid.clone().requires_grad_(True)
    log_prob = probs[0]*get_implicit_cls_lprob(grid,means,cov,pi,0)
    log_prob += probs[1]*get_implicit_cls_lprob(grid,means,cov,pi,1)

    # -- backward --
    th.autograd.backward(log_prob,th.ones_like(log_prob))
    return grid.grad

def get_noisy_params(means,cov,pi):

    error_m = 2.
    error_c = 0.
    error_p = 0.

    means = means + error_m*th.randn_like(means)
    cov = cov + error_c*th.randn_like(cov)**2
    pi = pi + error_p*th.randn_like(pi)**2
    pi = pi/th.sum(pi,-1)

    return means,cov,pi

def get_hard_cls_field(grid,means,cov,pi,k0=0):

    # -- compute log prob --
    eps = 1e-15
    grid = grid.clone().requires_grad_(True)
    log_prob = th.log(get_gaussian_prob(grid,means[k0],cov[k0])+eps)

    # -- backward --
    th.autograd.backward(log_prob,th.ones_like(log_prob))
    return grid.grad

def get_soft_cls_field(grid,means,cov,pi,probs):

    # -- compute log prob --
    eps = 1e-15
    grid = grid.clone().requires_grad_(True)
    log_prob = probs[0]*th.log(get_gaussian_prob(grid,means[0],cov[0])+eps)
    log_prob += probs[1]*th.log(get_gaussian_prob(grid,means[1],cov[1])+eps)

    # -- backward --
    th.autograd.backward(log_prob,th.ones_like(log_prob))
    return grid.grad

def mshape(tensor,ns):
    return tensor.reshape(ns,ns)

def main():

    # -=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #       Parameters
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-

    means = th.tensor([[1.,1.0],
                      [-1.,-1.0]])
    cov = 1.5*th.tensor([
        [[2.,0.],
         [0.,2.]],
        [[2.,0.],
         [0.,2.]]])
    pi = th.tensor([0.5,0.5])

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #      Classifier-Free Vector Fields
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    nsamples,ulim,llim = 50,2,-2
    ns = nsamples
    gmm_grid = get_grid(nsamples,ulim,llim)
    gmm_probs = get_gmm_prob(gmm_grid,means,cov*.1,pi)
    nsamples,ulim,llim = 20,2,0
    grid = get_grid(nsamples,ulim,llim)
    ihard_cls_field = get_hard_implicit_cls_field(grid,means,cov,pi,0)
    isoft_cls_field = get_soft_implicit_cls_field(grid,means,cov,pi,[0.8,0.2])
    gt_cls_field = get_hard_cls_field(grid,means,cov,pi,0)

    # -- plot --
    ginfo = {'wspace':0.05, 'hspace':0.0,
             "top":.9,"bottom":0.01,"left":0.01,"right":0.99}
    fig,ax = plt.subplots(1,4,figsize=(12,3),gridspec_kw=ginfo)
    ax[0].pcolormesh(mshape(gmm_grid[:,0],ns),mshape(gmm_grid[:,1],ns),
                     mshape(gmm_probs,ns),shading='gouraud',cmap=plt.cm.Blues)
    rect = patches.Rectangle((0.,0.), 1.98, 1.98, linewidth=2,
                             edgecolor='r', facecolor='none')
    ax[0].add_patch(rect)

    u,v = gt_cls_field[:,0],gt_cls_field[:,1]
    ax[1].quiver(grid[:,0],grid[:,1],u,v)

    u,v = isoft_cls_field[:,0],isoft_cls_field[:,1]
    ax[2].quiver(grid[:,0],grid[:,1],u,v)

    u,v = ihard_cls_field[:,0],ihard_cls_field[:,1]
    ax[3].quiver(grid[:,0],grid[:,1],u,v)

    print("Soft: ",th.mean((isoft_cls_field - gt_cls_field)**2))
    print("Hard: ",th.mean((ihard_cls_field - gt_cls_field)**2))
    for i in range(len(ax)): ax[i].axis('off')
    ax[0].set_title(r"Gaussian Mixture Model")
    ax[1].set_title(r"Groundtruth")
    ax[2].set_title(r"Soft Classifier-Free Guidance")
    ax[3].set_title(r"Hard Classifier-Free Guidance")

    plt.savefig("vector_field_classifier_free.png",dpi=200)

if __name__ == "__main__":
    main()
