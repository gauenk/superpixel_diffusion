"""

   Generative Score Matching with Known
   Class Probabilities (i.e. P(S=s) is prob of class "s")
   samples a subspace of the original domain.


"""

import numpy as np
import torch as th
import scipy
from scipy.stats import gaussian_kde
from scipy.special import softmax

import seaborn as sns
import matplotlib.pyplot as plt


def viz_gaussian(ax,means,cov):

    # -- sample data --
    nsamples = 1000
    data = np.random.multivariate_normal(means, cov, nsamples)
    x, y = data.T

    # -- create meshgrid grid --
    nbins = 50
    k = gaussian_kde(data.T)
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # cmap = plt.cm.BuGn_r
    cmap = plt.cm.Blues
    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=cmap)
    ax.scatter(x,y,s=0.1)

def viz_gaussian_prob(ax,means,cov):

    # -- sample data --
    nsamples = 50
    smin,smax = -3,3
    xi,yi = np.mgrid[smin:smax:nsamples*1j, smin:smax:nsamples*1j]
    x,y = xi.ravel(),yi.ravel()
    states = np.stack([x,y]).T
    probs = compute_prob(states,means,cov)
    # probs = probs.reshape(50,50)

    # cmap = plt.cm.BuGn_r
    cmap = plt.cm.Blues
    ax.pcolormesh(xi, yi, probs.reshape(xi.shape), shading='gouraud', cmap=cmap)
    # ax.scatter(x,y,s=0.1)

def update_fxn(state,means,cov,inv_cov,pi,fixed_prob):
    probs = compute_prob(state,means,cov,pi)

    update = 0
    for k in range(len(pi)):
        update_k = update_fxn_gaussian(state,means[k],inv_cov[k],fixed_prob)
        update += pi[k] * update_k

    if not(fixed_prob is None):  # include KL term
        cluster_probs = compute_cluster_probs(state,means,cov,pi)
        for k in range(len(pi)):
            update_k = update_fxn_cluster(k,state,means,inv_cov,fixed_prob)
            update += (fixed_prob[k]-cluster_probs[:,[k]]) * update_k
        delta = np.abs(fixed_prob[None,] - cluster_probs)
        # print(np.mean(delta),delta.min(),delta.max())

    eps = 1e-15
    return update / (probs[:,None]+eps)

def update_fxn_gaussian(state,means,inv_cov,fixed_prob):
    delta = state - means[None,]
    mat = np.einsum('BNi,Bi ->BN', inv_cov[None,], delta)
    dim = inv_cov.shape[-2]
    const = np.power(2*np.pi,-dim/2.) * np.sqrt(np.linalg.det(inv_cov))
    rescale = np.exp(-1/2 * np.sum(delta * mat,-1))[:,None]
    update = const * rescale * (-mat)
    return update

def update_fxn_cluster(k,state,means,inv_cov,fixed_prob):
    # cluster_probs = compute_cluster_probs(state,means,cov,pi)
    delta = state - means[k][None,]
    mat = np.einsum('BNi,Bi ->BN', inv_cov[k][None,], delta)
    return -mat
    # print("k: ",k)
    # print(means)
    # print(state.shape)
    # print(means[k][None,].shape)
    # print(state)
    # print("."*10)
    # lam = 1
    # # print(state)
    # # print(state - means[k][None,])
    # # exit()
    # return -2*lam*(state - means[k][None,])

# def update_fxn_gaussian(state,means,inv_cov,fixed_prob):
#     delta = means[None,]-state
#     # print(delta.shape)
#     update0 = np.einsum('BNi,Bi ->BN', inv_cov[None,], delta)
#     # print(delta.shape,inv_cov.shape,update0.shape)
#     # exit()
#     # print(update0.shape)
#     # exit()
#     update1 = 0 # grad(DK(p(x @ final state)|p(x @ current state)))
#     update = update0 + update1
#     # print(update)
#     # exit()
#     return update

def sampling(means,cov,pi,fixed_prob):

    # -- params --
    dim = cov.shape[-1]
    nsamples = 30000
    # nsamples = 1000

    # -- init conv --
    inv_cov = []
    for k in range(len(pi)):
        inv_cov.append(np.linalg.pinv(cov[k]))

    # -- init state --
    state = np.zeros((nsamples,dim))
    for d in range(dim):
        # state[:,d] = np.random.uniform(low=-4,high=4,size=nsamples)
        state[:,d] = np.random.uniform(low=-2,high=2,size=nsamples)

    # alpha = 0.05
    nsteps = 3000
    ntrace = 100
    trace = np.zeros((ntrace,nsamples,dim))
    alpha0 = 1e-3
    beta = 2e-4
    for i in range(nsteps):
        noise = np.random.normal(size=(nsamples,dim))
        update = update_fxn(state,means,cov,inv_cov,pi,fixed_prob)
        update = np.clip(update,-1e6,1e6)

        alpha = alpha0 * np.exp(-beta*i)
        alpha_n = np.sqrt(2*alpha)
        state = state + alpha*update + alpha_n*noise
    return state,trace

def check_samples(means,cov,samples):
    est_means = np.mean(samples,axis=0)
    est_cov = np.cov(samples.T)
    print(means)
    print(cov)
    print(est_means.round(decimals=2))
    print(est_cov.round(decimals=2))

def compute_prob(state,means,cov,pi):
    prob = 0
    for k in range(len(pi)):
        prob += pi[k] * compute_prob_gaussian(state,means[k],cov[k])
    return prob

def compute_prob_gaussian(state,means,cov):
    dim = int(cov.shape[-1])
    inv_cov = np.linalg.pinv(cov)[None,]
    const = np.power(2*np.pi,-dim/2.) * 1./np.sqrt(np.linalg.det(cov))
    delta = state - means[None,]
    # print("state.shape,means.shape: ",state.shape,means.shape,delta.shape,inv_cov.shape)
    mat = np.einsum('BNi,Bi ->BN', inv_cov, delta)
    probs = np.exp(-1/2 * np.sum(delta * mat,-1))
    return const * probs

def compute_cluster_probs(state,means,cov,pi):
    nsamples = state.shape[0]
    sims = np.zeros((nsamples,len(pi)))
    for k in range(len(pi)):
        inv_cov = np.linalg.pinv(cov[k])
        delta = state - means[k][None,]
        mat = np.einsum('BNi,Bi ->BN', inv_cov[None,], delta)
        sims[:,k] = -np.sum(delta * mat,-1)
        # sims[:,k] = -lam*np.sum(np.power(state-means[k],2),1)
    sims = softmax(sims,1)
    # sims = sims/sims.sum()
    return sims

def filter_z(samples):
    fixed_z_value = -0.0
    args = np.where(np.abs(samples[:,2]-fixed_z_value)<1e-1)
    return samples[args]
    # print(samples.shape)
    # return samples

def main():

    pi = np.array([0.5,0.5])
    means = np.array([[0.0,0.0,1.0],
                      [0.0,-0.0,-1.0]])
    cov = np.array([
        [[1.5,0.,0.],
         [0.,1.5,0.],
         [0.,0.,1.]],
        [[0.75,0.,0.],
         [0.,0.75,0.],
         [0.,0.,1.]],
    ])
    state = np.array([[0.0,0.,1.0]])
    # means = means[:,:2]
    # cov = 2.*cov[:,:2,:2]
    # state = state[:,:2]
    # fixed_prob = compute_cluster_probs(state,means,cov,pi)[0]
    fixed_prob = None
    print("fixed_prob: ",fixed_prob)
    samples0,trace0 = sampling(means,cov,pi,fixed_prob)
    fixed_prob = compute_cluster_probs(state,means,cov,pi)[0]
    print("fixed_prob: ",fixed_prob)
    samples1,trace1 = sampling(means,cov,pi,fixed_prob)
    check_samples(means,cov,samples0)
    check_samples(means,cov,samples1)
    print(samples1.shape)

    probs0 = compute_cluster_probs(samples0,means,cov,pi)
    probs1 = compute_cluster_probs(samples1,means,cov,pi)
    print("probs0: ",probs0.mean(0))
    print("probs1: ",probs1.mean(0))

    # -- filter for viz --
    samples0_f = filter_z(samples0)
    samples1_f = filter_z(samples1)
    # samples0_f = samples0
    # samples1_f = samples1

    ginfo = {'wspace':0.0, 'hspace':0.0,
             "top":.9,"bottom":0.1,"left":0.05,"right":0.99}
    fig,ax = plt.subplots(1,4,figsize=(8,4),gridspec_kw=ginfo)
    # ax.plot(trace[-100:,0,0],trace[-100:,0,1],'y-x',markersize=0.1,linewidth=0.1)
    # viz_gaussian(ax,means,cov)
    # viz_gaussian_prob(ax,means,cov)
    # ax.scatter(trace[0,:,0],trace[0,:,1],s=0.2,c='orange')
    ax[0].scatter(samples0_f[:,0],samples0_f[:,1],s=0.2,c='red')
    ax[1].scatter(samples1_f[:,0],samples1_f[:,1],s=0.2,c='blue')
    ax[2].scatter(samples0[:,1],samples0[:,2],s=0.2,c='red')
    ax[3].scatter(samples1[:,1],samples1[:,2],s=0.2,c='blue')
    for i in range(len(ax)):
        ax[i].set_xlim([-4,4])
        ax[i].set_ylim([-4,4])

    plt.savefig("gaussian_ring.png",dpi=200)

if __name__ == "__main__":
    main()
