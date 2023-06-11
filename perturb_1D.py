import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess
import time
from joblib import Parallel, delayed
import itertools


def dispersion(kx,a=1,t=1):
    e_k=-2*t*np.cos(kx*a)
    return e_k

def G(sig,kx,iom,mu,a=1,t=1):
    return 1/(iom+mu-2*t*np.cos(kx*a)-sig)

def plot_G():
    sigma=np.loadtxt('Sig.OCA.30')[:500,:]
    return 0

def sig_k(beta,U,knum=10):
    n=500
    sigma=np.loadtxt('Sig.OCA.30')[:n,:]
    fermion_freq=1j*(2*np.arange(n)+1)*np.pi/beta
    boson_freq=1j*(2*np.arange(n)+2)*np.pi/beta
    pert_sig=np.zeros((n,knum))
    k1=np.linspace(-np.pi/a,np.pi/a,num=knum+1)
    k2=np.roll(k1,1)
    kave=(k1+k2)/2
    klist=kave[-knum:] # this klist is for k points. i.e. not k+q.
    kqlist=k1[-knum:]# this klist is for k+q points. i.e. not k.
    halfknum=int(knum/2)
    qlist=klist[-halfknum:]
    # print('qlist',qlist)
    # print('klist',klist)
    # print('kqlist',kqlist)
    for omind in np.arange(n):
        for Omind in np.arange(n):
            for kx in klist:
                for ky in klist:
                    