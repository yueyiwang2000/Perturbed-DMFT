import numpy as np
from perturb_lib import *
import time
import matplotlib.pyplot as plt
from numba import jit, complex128
import perturb_lib
from numba.types import float64, complex128
# experimental features. try to avoid tau=0 and tau=beta to get well-defined physical quantities
def G12_shift(G12,q,knum,opt):
    qx=q[0]
    qy=q[1]
    qz=q[2]
    kind_list = np.arange(knum)
    kxind, kyind, kzind = np.meshgrid(kind_list, kind_list, kind_list, indexing='ij')
    if opt==1:#with factor
        G_12_factor=(-1)**((np.mod(kxind + qx, knum)-(kxind+qx))/knum+(np.mod(kyind + qy, knum)-(kyind+qy))/knum+(np.mod(kzind + qz, knum)-(kzind+qz))/knum)
    else:
        G_12_factor=1.
    G12_kq = G_12_factor*G12[:, np.mod(kxind + qx, knum), np.mod(kyind + qy, knum), np.mod(kzind + qz, knum)]
    return G12_kq

def alpha_shift(alpha,q,knum):
    qx=q[0]
    qy=q[1]
    qz=q[2]
    kind_list = np.arange(knum)
    kxind, kyind, kzind = np.meshgrid(kind_list, kind_list, kind_list, indexing='ij')
    alpha_kq = alpha[ np.mod(kxind + qx, knum), np.mod(kyind + qy, knum), np.mod(kzind + qz, knum)]
    return alpha_kq

#-------fermion fft----------
def stupid_ft_fermion(Gk,n,taulist,fermion_om,k):
    Gktau=np.zeros(2*n,dtype=complex)
    for i in np.arange(2*n):
        tau=taulist[i]
        Gktau[i]=np.sum(Gk[:,k[0],k[1],k[2]]*np.exp(-1j*fermion_om*tau))
    return Gktau

def stupid_ift_fermion(Gktau,n,beta,fermion_om):
    G_iom=np.zeros(4*n,dtype=complex)
    taulist=np.linspace(0,beta,num=4*n+1)[:4*n]
    for i in np.arange(4*n):
        tau=taulist[i]
        G_iom[i]=np.sum(Gktau*np.exp(1j*fermion_om*tau))
    return G_iom

def fast_ft_fermion(Gk):
    N=np.shape(Gk)[0]
    # Gktau=np.fft.fft(Gk,axis=0)*np.exp(1j*(N-1)*np.pi*np.arange(N)/N-1j*(2*np.arange(N)-N+1)*np.pi*0.5/N)[:,None,None,None]
    Gktau=np.fft.fft(Gk*np.exp(-1j*(2*np.arange(N)-N+1)*np.pi*0.5/N)[:,None,None,None],axis=0)*np.exp(1j*(N-1)*np.pi*np.arange(N)/N)[:,None,None,None]
    return Gktau

def fast_ift_fermion(Gk):# same way back. in fft, we fft then shift; in ifft, we shift back then fft.
    N=np.shape(Gk)[0]
    Gkiom=np.fft.ifft(Gk*np.exp(-1j*(N-1)*np.pi*np.arange(N)/N)[:,None,None,None],axis=0)*np.exp(+1j*(2*np.arange(N)-N+1)*np.pi*0.5/N)[:,None,None,None]
    return Gkiom

#-------boson fft----------
# here i use stupid version because I want N+1 freq points.... may be this is not necessary but this is conceptially clear and only take a little time.
# THIS IS REALLY SLOW! rewrite it when have time, use the basic idea of fft.
# HERE I just use numba.... I am lazy. It's faster than the most stupid numpy version.
@jit(complex128[:](complex128[:], float64, float64[:]), nopython=True)
def stupid_ift_boson(Pq_tau, beta, boson_om):
    N = len(Pq_tau)
    P_iom = np.zeros_like(boson_om, dtype=complex128)
    taulist = (np.arange(N)+0.5) / N * beta
    for i in range(len(boson_om)):
        sum_val = 0.0 + 0.0j
        for j in range(N):
            sum_val += Pq_tau[j] * np.exp(1j * boson_om[i] * taulist[j])
        P_iom[i] = sum_val / N 
    return P_iom

#-----------convolution-----------
# This is only for G12, which means, for Green's functions that have well-defined fourier transformations.
def precalcP_fft(q, knum, n, Gk,beta,opt):# this function deal with Gk*Gkq. they should be well sliced and shifted.
    N=2*n
    fermion_om = (2*np.arange(2*n)+1-2*n)*np.pi/beta
    boson_om = (2*np.arange(n+1))*np.pi/beta
    Gk_tau=fast_ft_fermion(Gk)
    Gkq_tau=G12_shift(Gk_tau,q,knum,opt)
    Pq_tau=np.sum(-Gk_tau[::-1,:,:,:]*Gkq_tau,axis=(1,2,3))/knum**3/beta
    return Pq_tau.real

# this is for those function which has ill-defined Green's functions. like 1/iom scaling.
def precalcP_fft_diag(q, knum, n, Gk,beta,delta_inf,alpha_k):
    N=2*n
    alpk=alpha_k[None,:,:,:]
    tlist=(np.arange(N)+0.5)/N*beta
    fermion_om = (2*np.arange(N)+1-N)*np.pi/beta
    # boson_om = ((2*np.arange(n+1)-n)*np.pi/beta)
    # calculating Gk+q(tau)
    Gk0=1/2*((1+delta_inf/alpk)/(1j*fermion_om[:,None,None,None]-alpk)+
             (1-delta_inf/alpk)/(1j*fermion_om[:,None,None,None]+alpk))
    Gk_tau_diff=fast_ft_fermion(Gk-Gk0)
    Gk_tau_ana=-beta/2*((1+delta_inf/alpk)*np.exp(-alpk*tlist[:,None,None,None])/(1+np.exp(-alpk*beta))+
                        (1-delta_inf/alpk)*np.exp(alpk*tlist[:,None,None,None])/(1+np.exp(alpk*beta)))
    Gk_tau=Gk_tau_ana+Gk_tau_diff
    Gkq_tau=G12_shift(Gk_tau,q,knum,0)
    Pq_tau=np.sum(-Gk_tau[::-1,:,:,:]*Gkq_tau,axis=(1,2,3))/knum**3/beta
    return Pq_tau.real

def precalcsig_fft(q, knum, Gk,Pq_tau,beta,U,opt):#for off-diagonal
    N=np.shape(Pq_tau)[0]
    Gk_tau=fast_ft_fermion(Gk)
    Gkq_tau=G12_shift(Gk_tau,q,knum,opt)
    sig_tau=np.sum(Pq_tau*Gkq_tau,axis=(1,2,3))*(-1)*U**2/knum**3/beta
    sig_iom=np.fft.ifft(sig_tau*np.exp(-1j*(N-1)*np.pi*np.arange(N)/N))*np.exp(+1j*(2*np.arange(N)-N+1)*np.pi*0.5/N)
    return sig_iom


def precalcsig_fft_diag(q, knum, Gk,Pq_tau,beta,U,delta_inf,alpha_k):
    N=np.shape(Pq_tau)[0]
    alpk=alpha_k[None,:,:,:]
    fermion_om = (2*np.arange(N)+1-N)*np.pi/beta
    tlist=np.arange(N)/N*beta
    Gk0=1/2*((1+delta_inf/alpk)/(1j*fermion_om[:,None,None,None]-alpk)+
             (1-delta_inf/alpk)/(1j*fermion_om[:,None,None,None]+alpk))
    Gk_tau_ana=-beta/2*((1+delta_inf/alpk)*np.exp(-alpk*tlist[:,None,None,None])/(1+np.exp(-alpk*beta))+
                        (1-delta_inf/alpk)*np.exp(alpk*tlist[:,None,None,None])/(1+np.exp(alpk*beta)))
    Gk_tau_diff=fast_ft_fermion(Gk-Gk0)
    Gkq_tau_diff=G12_shift(Gk_tau_diff,q,knum,0)
    Gkq_tau_ana=G12_shift(Gk_tau_ana,q,knum,0)
    Gkq_tau=Gkq_tau_diff+Gkq_tau_ana
    sig_tau=np.sum(Pq_tau*Gkq_tau,axis=(1,2,3))*(-1)*U**2/knum**3/beta
    sig_iom=np.fft.ifft(sig_tau*np.exp(-1j*(N-1)*np.pi*np.arange(N)/N))*np.exp(+1j*(2*np.arange(N)-N+1)*np.pi*0.5/N)
    return sig_iom

# brute-force method.
def precalcP_bf(q, knum, n, G12,beta,opt):
    # start_time = time.time()
    kind_list = np.arange(knum)
    kxind, kyind, kzind = np.meshgrid(kind_list, kind_list, kind_list, indexing='ij')
    qx=q[0]
    qy=q[1]
    qz=q[2]
    # q_time = time.time()
    P_partial = np.zeros((n+1),dtype=np.complex128)
    if opt==1:
        G_12_factor=(-1)**((np.mod(kxind + qx, knum)-(kxind+qx))/knum+(np.mod(kyind + qy, knum)-(kyind+qy))/knum+(np.mod(kzind + qz, knum)-(kzind+qz))/knum)
    else:
        G_12_factor=1.
    # factor_time = time.time()
    G12_kq = G_12_factor*G12[:, np.mod(kxind + qx, knum), np.mod(kyind + qy, knum), np.mod(kzind + qz, knum)]
    G12_sliced = G12[int(n/2):int(3*n/2)]
    # slice_time=time.time()
    for Omind in np.arange(n+1):
        G12_kq_sliced=G12_kq[Omind:n+Omind]
        Gmul=G12_sliced * G12_kq_sliced# takes most time
        P_partial[Omind] = np.sum(Gmul)#.real
        # P_partial[2*n-Omind]=P_partial[Omind]   
    # end_time = time.time()
    # print('slicing time(brute-force):',slice_time-start_time)
    # print('calculation time(brute-force):',end_time-slice_time)
    return P_partial/ beta * (1 / knum) ** 3


#-------------test---------------
def conv_test(sigA,sigB,U,T,knum):
    mu=U/2
    beta=1/T
    z_A=perturb_lib.z(beta,mu,sigA)
    z_B=perturb_lib.z(beta,mu,sigB)
    n=sigA.size
    N=2*n
    allsigA=perturb_lib.ext_sig(beta,sigA)
    allsigB=perturb_lib.ext_sig(beta,sigB)
    G11=perturb_lib.G_11(knum,z_A,z_B)
    G12=perturb_lib.G_12(knum,z_A,z_B)
    qtest=[1,2,3]

    #preparation for this trick.
    k1,k2,k3=gen_full_kgrids(knum)
    delta_inf=np.abs(-mu+allsigA[-1].real)
    alphak=np.sqrt(dispersion(k1,k2,k3)**2+delta_inf**2)
    # f_alphak=fermi(alphak,beta)
    # opt=1
    # p11_fft=precalcP_fft(qtest,knum,n,G12,beta,opt)[0:int(n/2)]
    # p11_bf=precalcP_bf(qtest,knum,n,G12,beta,opt)[int(n/2):]
    

    opt=0
    p11_fft=precalcP_fft_diag(qtest,knum,n,G11,beta,delta_inf,alphak)
    p11_bf=precalcP_bf(qtest,knum,n,G11,beta,opt)

    plt.plot(p11_fft.real,label='fft.real')
    plt.plot(p11_fft.imag,label='fft.imag')
    plt.plot(p11_bf.real,label='bf.real')
    plt.plot(p11_bf.imag,label='bf.imag')
    plt.legend()
    plt.show()
    return 0

if __name__ == "__main__":
    T=0.37
    U=7.0
    knum=10
    nfreq=500
    sigma=np.loadtxt('./trial_sigma/{}_{}.dat'.format(U,T))[:nfreq,:]
    sigA=sigma[:,1]+1j*sigma[:,2]#sig+delta
    sigB=sigma[:,3]+1j*sigma[:,4]#sig-delta
    conv_test(sigA,sigB,U,T,knum)