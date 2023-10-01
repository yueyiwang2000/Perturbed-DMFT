#!/usr/bin/env python
import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess
# here we assume the sigma has 5 columns.
"""
This file contains useful functions to examine various output files.
"""

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


#single mode
def single_mode(num):
    ilist=np.arange(num)
    for i in ilist:
        filename='./'
        # filename='./files/{}_{}/Delta.OCA.{}'.format(U,T,int(i+1))
        # filename='./files/{}_{}/Sig.pert.dat.{}'.format(U,T,int(i+1))
        filename='./files_pert_boldc/{}_{}/sig12.dat.{}'.format(U,T,int(i+1))
        plot_sigma(filename,i)
        plt.legend()
        plt.title(filename)
        plt.show()

def stable_test(num):
    ilist=np.arange(num)
    # ilist=np.array([0,2])
    for i in ilist:
        # filename='./files/{}_{}/Sig.OCA.{}'.format(U,T,int(i+1))
        # filename='./files/{}_{}/Delta.OCA.{}'.format(U,T,int(i+1))
        filename='./files/{}_{}/Sig.out.{}'.format(U,T,int(i))
        sigma=np.loadtxt(filename)
        omega=sigma[:,0]
        plt.plot(omega,sigma[:,1],label='perturbed DMFT self-energy {}'.format(i),color='red')

        filename='./files/{}_{}/ori_Sig.out.{}'.format(U,T,int(i))
        sigma=np.loadtxt(filename)
        omega=sigma[:,0]
        plt.plot(omega,sigma[:,1],label='original DMFT self-energy {}'.format(i),color='blue')
    filename='./trial_sigma/{}_{}.dat'.format(U,T)
    sigma=np.loadtxt(filename)
    omega=sigma[:,0]
    plt.plot(omega,sigma[:,1],label='original sigma we start with',color='green')
    plt.legend()
    plt.show()

def compare_sig(ind):
    freq_num=500
    if mode ==1:
        filename='./files_pert_ctqmc/{}_{}/Sig.out.{}'.format(U,T,ind)
    elif mode==0:
        filename='./files_pert_boldc/{}_{}/Sig.OCA.{}'.format(U,T,ind+1)
    sigma=np.loadtxt(filename)
    omega=sigma[:freq_num,0]
    plt.plot(omega,sigma[:freq_num,1],label='1st column after perturbed DMFT')
    plt.plot(omega,sigma[:freq_num,2],label='2nd column after perturbed DMFT')
    plt.plot(omega,sigma[:freq_num,3],label='3rd column after perturbed DMFT')
    plt.plot(omega,sigma[:freq_num,4],label='4th column after perturbed DMFT')
    if mode ==1:
        filename='./files_pert_ctqmc/{}_{}/Sig.out.{}'.format(U,T,ind)
    elif mode==0:
        filename='./files_pert_boldc/{}_{}/ori_Sig.OCA.{}'.format(U,T,ind+1)
    sigma=np.loadtxt(filename)
    omega=sigma[:freq_num,0]
    plt.plot(omega,sigma[:freq_num,1],label='1st column after DMFT')
    plt.plot(omega,sigma[:freq_num,2],label='2nd column after DMFT')
    plt.plot(omega,sigma[:freq_num,3],label='3rd column after DMFT')
    plt.plot(omega,sigma[:freq_num,4],label='4th column after DMFT')
    # if ind>=1:
    #     # filename='./files_pert_ctqmc/{}_{}/pert_Sig.out.{}'.format(U,T,ind-1)
    #     filename='./files_pert_boldc/{}_{}/Sig.OCA.{}'.format(U,T,ind)
    #     sigma=np.loadtxt(filename)
    #     omega=sigma[:freq_num,0]
    #     plt.plot(omega,sigma[:freq_num,1],label='1th column in sigma we start with')
    #     plt.plot(omega,sigma[:freq_num,3],label='3rd column in sigma we start with')
    plt.legend()
    plt.show()

def compare_Delta(ind):
    if mode ==1:
        filename='./files_pert_ctqmc/{}_{}/pert_Delta.inp.{}'.format(U,T,ind)
    elif mode==0:
        filename='./files_pert_boldc/{}_{}/Delta.inp.{}'.format(U,T,ind+1)
    sigma=np.loadtxt(filename)
    omega=sigma[:,0]
    plt.plot(omega,sigma[:,1],label='11 real pert_Delta.inp')
    plt.plot(omega,sigma[:,2],label='11 imag pert_Delta.inp')
    plt.plot(omega,sigma[:,3],label='22 real pert_Delta.inp')
    plt.plot(omega,sigma[:,4],label='22 imag pert_Delta.inp')
    if mode ==1:
        filename='./files_ctqmc/{}_{}/ori_Delta.inp.{}'.format(U,T,ind)
    if mode ==0:
        filename='./files_boldc/{}_{}/ori_Delta.inp.{}'.format(U,T,ind+1)
    sigma=np.loadtxt(filename)
    omega=sigma[:,0]
    plt.plot(omega,sigma[:,1],label='11 real Delta.inp')
    plt.plot(omega,sigma[:,2],label='11 imag Delta.inp')
    plt.plot(omega,sigma[:,3],label='22 real Delta.inp')
    plt.plot(omega,sigma[:,4],label='22 imag Delta.inp')
    plt.legend()
    plt.show()

def burst_scatter_sig(num):
    ilist=np.arange(num)
    for i in ilist:
        if mode==0:
            filename='./files_boldc/{}_{}/ori_Sig.OCA.{}'.format(U,T,int(i+1))
        elif mode==1:
            filename='./files_ctqmc/{}_{}/ori_Sig.out.{}'.format(U,T,int(i))
        
        if os.path.isfile(filename):
            sigma=np.loadtxt(filename)
            # omega=sigma[:,0]
            plt.scatter(i,sigma[-1,1],c='red')
            plt.scatter(i,sigma[-1,3],c='red')
        else:
            print('cannot find {}'.format(filename))
        if mode ==0:
            filename='./files_pert_boldc/{}_{}/Sig.OCA.{}'.format(U,T,int(i+1))
        elif mode ==1:
            filename='./files_pert_ctqmc/{}_{}/Sig.out.{}'.format(U,T,int(i))
        if os.path.isfile(filename):
            sigma=np.loadtxt(filename)
            # omega=sigma[:,0]
            plt.scatter(i,sigma[-1,1],c='blue')
            plt.scatter(i,sigma[-1,3],c='blue')
        else:
            print('cannot find {}'.format(filename))
    plt.title('Sigma_imp(om->inf).real U={},T={}'.format(U,T))
    plt.xlabel('DMFT iterations Red:DMFT Blue:DMFT+pert')
    plt.show()
    return 0

if __name__ == "__main__":
    mode=1
    #boldc=0, ctqmc=1
    U=7.0
    T=0.38
    # print("format: plot_sigma.py U T")
    burst_scatter_sig(20)
    # compare_Delta(6)
    # single_mode(10)