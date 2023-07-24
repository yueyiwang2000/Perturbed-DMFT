#!/usr/bin/env python
import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess
# here we assume the sigma has 5 columns.


def plot_sigma(filename,index):
    sigma=np.loadtxt(filename)
    omega=sigma[:,0]
    # for i in np.arange(np.size(sigma[1,:])-1)+1:
    #     plt.plot(omega,sigma[:,i],label='{}th column in {}th file'.format(i,index))
    plt.plot(omega,sigma[:,1],label='real1 in {}th file'.format(index))
    # plt.plot(omega,sigma[:,3],label='real2 in {}th file'.format(index))
    # plt.plot(omega,np.zeros_like(omega),label='ZERO')
    plt.legend()
    # plt.xlim((0,omega[-1]/10))
    # plt.ylim((-0.2,0.2))


U=7.0
T=0.39
#single mode
def single_mode(num):
    ilist=np.arange(num)
    for i in ilist:
        filename='./'
        # filename='./files/{}_{}/Delta.OCA.{}'.format(U,T,int(i+1))
        # filename='./files/{}_{}/Sig.pert.dat.{}'.format(U,T,int(i+1))
        
        plot_sigma(filename,i)
        plt.legend()
        plt.title(filename)
        plt.show()

def burst_mode(num):
    ilist=np.arange(num)
    # ilist=np.array([0,2])
    for i in ilist:
        # filename='./files/{}_{}/Sig.OCA.{}'.format(U,T,int(i+1))
        # filename='./files/{}_{}/Delta.OCA.{}'.format(U,T,int(i+1))
        filename='./files/{}_{}/Sig.out.{}'.format(U,T,int(i))
        plot_sigma(filename,i)
    plt.legend()
    plt.show()

def compare_sig(ind):
    freq_num=500
    filename='./files/{}_{}/Sig.out.{}'.format(U,T,ind)
    sigma=np.loadtxt(filename)
    omega=sigma[:freq_num,0]
    plt.plot(omega,sigma[:freq_num,1],label='1th column after perturbed DMFT')
    plt.plot(omega,sigma[:freq_num,3],label='3rd column after perturbed DMFT')

    filename='./files/{}_{}/ori_Sig.out.{}'.format(U,T,ind)
    sigma=np.loadtxt(filename)
    omega=sigma[:freq_num,0]
    plt.plot(omega,sigma[:freq_num,1],label='1th column after DMFT')
    plt.plot(omega,sigma[:freq_num,3],label='3rd column after DMFT')
    
    # filename='./trial_sigma/{}_{}.dat'.format(U,T)
    # sigma=np.loadtxt(filename)
    # omega=sigma[:freq_num,0]
    # plt.plot(omega,sigma[:freq_num,1],label='1th column in original sigma we start with')
    # plt.plot(omega,sigma[:freq_num,3],label='3rd column in original sigma we start with')
    plt.legend()
    plt.show()

def compare_Delta(ind):
    filename='./files/{}_{}/pert_Delta.inp.{}'.format(U,T,ind)
    sigma=np.loadtxt(filename)
    omega=sigma[:,0]
    # plt.plot(omega,sigma[:,1],label='1th column in Delta.inp.0')
    plt.plot(omega,sigma[:,2],label='2th column in pert_Delta.inp.0')
    # plt.plot(omega,sigma[:,3],label='3rd column in Delta.inp.0')
    plt.plot(omega,sigma[:,4],label='4th column in pert_Delta.inp.0')

    filename='./files/{}_{}/ori_Delta.inp.{}'.format(U,T,ind)
    sigma=np.loadtxt(filename)
    omega=sigma[:,0]
    # plt.plot(omega,sigma[:,1],label='1th column in ori_Delta.inp.0')
    plt.plot(omega,sigma[:,2],label='2th column in ori_Delta.inp.0')
    # plt.plot(omega,sigma[:,3],label='3rd column in ori_Delta.inp.0')
    plt.plot(omega,sigma[:,4],label='4th column in ori_Delta.inp.0')
    plt.legend()
    plt.show()


def burst_scatter(num):
    ilist=np.arange(num)
    for i in ilist:
        filename='./files/{}_{}/Sig.out.{}'.format(U,T,int(i))
        # filename='./files/{}_{}/Delta.out.{}'.format(U,T,int(i))
        if os.path.isfile(filename):
            sigma=np.loadtxt(filename)
            # omega=sigma[:,0]
            plt.scatter(i,sigma[-1,1],label='real1',c='blue')
            plt.scatter(i,sigma[-1,3],label='real1',c='red')
        else:
            print('cannot find {}'.format(filename))
    # plt.legend()
    plt.show()
    return 0
# 
# single_mode(1)
# burst_mode(12)
# compare_sig()
# compare_Delta()
# burst_scatter(10)
for ind in np.arange(10):
    compare_sig(ind)
#     compare_Delta(ind)