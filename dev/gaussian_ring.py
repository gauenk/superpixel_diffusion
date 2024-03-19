"""

   Generative Score Matching with Known
   Class Probabilities (i.e. P(S=s) is prob of class "s")
   samples a subspace of the original domain.


"""

import numpy as np
import torch as th
import scipy
from scipy.stats import gaussian_kde

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

def update_fxn(state,means,inv_cov,fixed_prob):
    delta = means-state
    update0 = np.einsum('BNi,Bi ->BN', inv_cov, delta)
    update1 = 0 # grad(DK(p(x @ final state)|p(x @ current state)))
    update = update0 + update1
    return update

def sampling(means,cov,fixed_prob):

    nsamples = 1000
    means = means[None,:]
    inv_cov = np.linalg.pinv(cov)
    inv_cov = inv_cov[None,:]

    # state = np.repeat(means.copy(),nsamples,axis=0).copy()
    state = np.zeros((nsamples,2))
    state[:,0] = np.random.uniform(low=-3,high=3,size=nsamples)
    state[:,1] = np.random.uniform(low=-3,high=3,size=nsamples)

    alpha = 0.05
    alpha_n = np.sqrt(2*alpha)
    nsteps = 10000
    trace = [state.copy()]
    for i in range(nsteps):
        noise = np.random.normal(size=(nsamples,2))
        update = update_fxn(state,means,inv_cov,fixed_prob)
        state = state + alpha*update + alpha_n*noise
        trace.append(state)
    trace = np.array(trace)
    return state,trace

def check_samples(means,cov,samples):
    est_means = np.mean(samples,axis=0)
    est_cov = np.cov(samples.T)
    print(est_means)
    print(est_cov)

def compute_prob(state,means,cov):
    inv_cov = np.linalg.pinv(cov)[None,]
    const = 1./(2*np.pi) * 1./np.sqrt(np.linalg.det(cov))
    delta = state - means
    mat = np.einsum('BNi,Bi ->BN', inv_cov, delta)
    # print(delta.shape,mat.shape)
    # print(np.sum(delta * mat,-1).shape)
    probs = np.exp(-1/2 * np.sum(delta * mat,-1))
    return const * probs

def main():

    means = np.array([0,0])
    cov = np.array([[1.,0.],[0.,4.]])
    state = np.array([[0.75,0.75]])
    fixed_prob = compute_prob(state,means,cov)[0]
    print(fixed_prob)
    samples,trace = sampling(means,cov,fixed_prob)
    # check_samples(means,cov,samples)

    ginfo = {'wspace':0.0, 'hspace':0.0,
             "top":.9,"bottom":0.1,"left":0.1,"right":0.99}
    fig,ax = plt.subplots(1,1,figsize=(4,4),gridspec_kw=ginfo)
    # ax.plot(trace[-100:,0,0],trace[-100:,0,1],'y-x',markersize=0.1,linewidth=0.1)
    # viz_gaussian(ax,means,cov)
    viz_gaussian_prob(ax,means,cov)
    # ax.scatter(trace[0,:,0],trace[0,:,1],s=0.2,c='orange')
    # ax.scatter(samples[:,0],samples[:,1],s=0.2,c='red')

    plt.savefig("gaussian_ring.png",dpi=200)

if __name__ == "__main__":
    main()
