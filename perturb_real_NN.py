import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess
import time
from joblib import Parallel, delayed

#------------------------Yueyi Wang. Jun 16 2023----------------#
# This code is written as an simple demo of perturbed DMFT for Hubbard model on a 3D AFM system with 
# a complex unit cell with 2 orbitals. In this code I used real-space summation which doesn't
# require real-space summation because the 'locality' of Hubbard U.
# To avoid some technical issue and simplify the question (this is only a demo),
# only NNs (nearest neighbours) are considered.


def dispersion(kx,ky,kz,a=1,t=1):
    e_k=-2*t*np.cos(kx*a)-2*t*np.cos(ky*a)-2*t*np.cos(kz*a)
    return e_k


def foldback(k,knum):# from complete 2*knum k points per dimension to a reduced knum k points.
    return np.where((k>=0)&(k<knum),k,2*knum-k-1)# 
    
def gen_qindices(qlist):
    qindices=[]
    for i in qlist:
        for j in qlist:
            for k in qlist:
                if i<=j<=k:
                    qindices+=[(i,j,k)]
    # print(qindices)
    return qindices


def kgrid_full(knum,a=1):
    k1=np.linspace(-np.pi/a,np.pi/a,num=knum+1)
    k2=np.roll(k1,1)
    kave=(k1+k2)/2
    klist=kave[-knum:] 
    # print('klist=',klist)
    k1, k2, k3 = np.meshgrid(klist, klist, klist, indexing='ij')
    kx=0.5*(-k1+k2+k3)
    ky=0.5*(k1-k2+k3)
    kz=0.5*(k1+k2-k3)
    return kx,ky,kz


# Use this, we want to generate an array G_sk(iomega) as a function of iomega.
# instead we prepare alpha and then generate Gf in the polarization
def z(beta,mu,sig):
    # sometimes we want values of G beyond the range of n matsubara points. try to do a simple estimation for even higher freqs:
    n=sig.size
    om=(2*np.arange(4*n)+1-4*n)*np.pi/beta
    allsig=ext_sig(beta,sig)
    z=om*1j+mu-allsig
    return z

def z0(beta,mu,allsig):# Here, alpha=i*omega+mu-sig(inf).real
    n=int(allsig.size/4)
    om=(2*np.arange(4*n)+1-4*n)*np.pi/beta
    # allsig=ext_sig(beta,sig)
    z0=om*1j+mu-allsig[-1].real#
    return z0

def ext_sig(beta,sig):
    lenom=sig.size
    # print(lenom)
    all_om=(2*np.arange(2*lenom)+1)*np.pi/beta
    allsig=np.zeros(4*lenom,dtype=complex)
    allsig[2*lenom:3*lenom]=sig
    allsig[3*lenom:4*lenom]=sig[lenom-1].real+1j*sig[lenom-1].imag*all_om[lenom-1]/all_om[lenom:2*lenom]
    allsig[:2*lenom]=allsig[4*lenom:2*lenom-1:-1].conjugate()
    return allsig

def G_offdiag(knum,z_A,z_B,a=1):
    n=z_A.size
    kx,ky,kz=kgrid_full(knum)
    G_offdiag=np.zeros((n,knum,knum,knum),dtype=np.complex128)
    zazb=z_A*z_B
    dis=dispersion(kx, ky, kz)
    G_offdiag = dis / (zazb[:, None, None, None] - dis**2)
    return G_offdiag

def FT_3D(knum,G_k,Rx,Ry,Rz):# usually R is like (0,0,1)
    n=np.shape(G_k)[0]
    G_real=np.zeros(n,dtype=complex)
    kx,ky,kz=kgrid_full(knum)
    exp_factor=np.exp(-1j*(kx*Rx+ky*Ry+kz*Rz))
    G_real=np.sum(G_k*exp_factor,axis=(1,2,3))/knum**3
    return G_real


#--------------test functions below---------

def G_test(a=1):
    # start_time = time.time()
    U=2.0
    mu=U/2
    T=0.01
    beta=1/T
    sigma=np.loadtxt('{}_{}.dat'.format(U,T))[:500,:]
    sigA=sigma[:,1]+1j*sigma[:,2]
    sigB=sigma[:,3]+1j*sigma[:,4]
    z_A=z(beta,mu,sigA)
    z_B=z(beta,mu,sigB)
    n=sigA.size
    # allsigA=ext_sig(beta,sigA)
    # print('n=',n,)
    knum=20
    G_off=G_offdiag(knum,z_A,z_B)
    fermion_om = (2*np.arange(4*n)+1-4*n)*np.pi/beta
    # time_G=time.time()
    # print("time to calculate prepare 2 G is {:.6f} s".format(time_G-start_time))
    kxind=3
    kyind=4
    kzind=5
    plt.plot(fermion_om,G_off[:,kxind,kyind,kzind].real,label='Gk_off_real')
    plt.plot(fermion_om,G_off[:,kxind,kyind,kzind].imag,label='Gk_off_imag')
    # plt.plot(fermion_om,G_off[:,knum-1-kxind,kyind,kzind].real,label='G-k_off_real')
    # plt.plot(fermion_om,G_off[:,knum-1-kxind,kyind,kzind].imag,label='G-k_off_imag')
    plt.legend()
    plt.show()
    #-------------FT test--------------
    Greal_100=FT_3D(knum,G_off,1,0,0)
    Greal_000=FT_3D(knum,G_off,0,0,0)
    plt.plot(fermion_om,Greal_100.real,label='Gr100_off_real')
    plt.plot(fermion_om,Greal_100.imag,label='Gr100_off_imag')
    plt.plot(fermion_om,Greal_000.real,label='Gr000_off_real')
    plt.plot(fermion_om,Greal_000.imag,label='Gr000_off_imag')
    plt.legend()
    plt.show()
    return 0

G_test()