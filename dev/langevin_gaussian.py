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
    nsamples = 10000
    data = np.random.multivariate_normal(means, cov, nsamples)
    x, y = data.T

    # -- create meshgrid grid --
    nbins = 70
    k = gaussian_kde(data.T)
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # cmap = plt.cm.BuGn_r
    cmap = plt.cm.Blues
    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=cmap)
    x = x[:1000]
    y = y[:1000]
    ax.scatter(x,y,s=0.1)

def update_fxn(state,means,inv_cov):
    delta = state-means
    update = np.einsum('BNi,Bi ->BN', inv_cov, delta)
    # print("update.shape: ",update.shape)
    return -update # -(inv_cov @ (state - means))

def sampling(means,cov):

    inv_cov = np.linalg.pinv(cov)
    nsamples = 1000
    means = means[None,:]
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
        state = state + alpha*update_fxn(state,means,inv_cov) + alpha_n*noise
        trace.append(state)

    trace = np.array(trace)
    return state,trace

def check_samples(means,cov,samples):
    est_means = np.mean(samples,axis=0)
    est_cov = np.cov(samples.T)
    print(est_means)
    print(est_cov)

def main():

    means = np.array([0,0])
    cov = np.array([[1.,0.],[0.,4.]])
    samples,trace = sampling(means,cov)
    check_samples(means,cov,samples)

    ginfo = {'wspace':0.0, 'hspace':0.0,
             "top":.9,"bottom":0.1,"left":0.1,"right":0.99}
    fig,ax = plt.subplots(1,1,figsize=(4,4),gridspec_kw=ginfo)
    # ax.plot(trace[-100:,0,0],trace[-100:,0,1],'y-x',markersize=0.1,linewidth=0.1)
    viz_gaussian(ax,means,cov)
    # ax.scatter(trace[0,:,0],trace[0,:,1],s=0.2,c='orange')
    # ax.scatter(samples[:,0],samples[:,1],s=0.2,c='red')
    ax.set_xlim([-6,6])
    ax.set_ylim([-6,6])

    plt.savefig("langevin_gaussian.png",dpi=200)

if __name__ == "__main__":
    main()
