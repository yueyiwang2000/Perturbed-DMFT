import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess,math
import time
import fft_convolution as convlib
import perturb_lib as lib
import perturb
'''
This file is a simple check of simple perturbation expansion of Hubbard Model.
This has nothing todo with DMFT
We use a simple form of Green's function and work in imaginary time domain.
FFT is done as a final step to see Sigma(iom).
'''
def singlecell_G11_tau(beta,eps,taulist):
    '''
    beta:1/T
    eps:dispersion in knum*knum*knum array
    taulist: tau points
    G11(iom)=1/(iom-eps)
    This is GF for single unit cell PM case
    '''
    return -beta*lib.fermi(eps[None,:,:,:],beta)*np.exp((beta-taulist)[:,None,None,None]*eps[None,:,:,:])



def PM_G11_tau(beta,eps,taulist):
    '''
    beta:1/T
    eps:dispersion in knum*knum*knum array
    taulist: tau points
    G11(iom)=(iom)/((iom)**2-eps**2)
    '''
    return -0.5*beta*lib.fermi(eps[None,:,:,:],beta)*(np.exp((beta-taulist)[:,None,None,None]*eps[None,:,:,:])+np.exp(taulist[:,None,None,None]*eps[None,:,:,:]))


def PM_G12_tau(beta,eps,taulist):
    '''
    beta:1/T
    eps:dispersion in knum*knum*knum array
    taulist: tau points
    G12(iom)=eps_k/((iom)**2-eps**2)
    '''
    return -0.5*beta*lib.fermi(eps[None,:,:,:],beta)*(np.exp((beta-taulist)[:,None,None,None]*eps[None,:,:,:])-np.exp(taulist[:,None,None,None]*eps[None,:,:,:]))


def AFM_G11_tau(beta,eps,taulist,delta):
    '''
    beta:1/T
    eps:dispersion in knum*knum*knum array
    taulist: tau points
    This is for AFM case.
    G11(iom)=(iom-delta)/((iom)**2-delta**2-eps**2)
    '''
    alpha=np.sqrt(delta**2+eps**2)[None,:,:,:]
    return -0.5*beta*lib.fermi(alpha,beta)*(np.exp((beta-taulist)[:,None,None,None]*alpha)*(1-delta/alpha)+np.exp(taulist[:,None,None,None]*alpha)*(1+delta/alpha))


def AFM_G12_tau(beta,eps,taulist,delta):
    '''
    beta:1/T
    eps:dispersion in knum*knum*knum array
    taulist: tau points
    This is for AFM case.
    G12(iom)=eps_k/((iom)**2-delta**2-eps**2)
    '''
    alpha=np.sqrt(delta**2+eps**2)[None,:,:,:]
    return -0.5*beta*eps[None,:,:,:]/alpha*lib.fermi(alpha,beta)*(np.exp((beta-taulist)[:,None,None,None]*alpha)-np.exp(taulist[:,None,None,None]*alpha))    



def check(knum,G_tau):
    max_sym_index,essential_kpoints, sym_array=lib.calc_sym_array(knum)
    for k in essential_kpoints[:pltkpts]:
        kxind=k[0]
        kyind=k[1]
        kzind=k[2]
        plt.plot(G_tau[:, kxind, kyind, kzind].real,label='G_tau real')
        plt.plot(G_tau[:, kxind, kyind, kzind].imag,label='G_tau imag')
        plt.grid()
        plt.legend()
        plt.show()
    return 0

def calc_P(G,knum,opt):
    '''
    opt=1: give a - sign when k->k+k1, where k1 is a reciprocal vector; others: do not care sign.
    '''
    Gtau1=G
    G1=-G[::-1,:,:,:]
    P=np.zeros_like(G)
    for q1 in np.arange(knum):
        for q2 in np.arange(knum):
            for q3 in np.arange(knum):
                q=[q1,q2,q3]
                Gtau2=convlib.G12_shift(G1,q,knum,opt)
                P[:,q1,q2,q3]=np.sum(Gtau1*Gtau2,axis=(1,2,3))
    return P/knum**3


def calc_Sig(P,G,knum,opt):
    '''
    opt=1: give a - sign when k->k+k1, where k1 is a reciprocal vector; others: do not care sign.
    '''
    Sig=np.zeros_like(G)
    for q1 in np.arange(knum):
        for q2 in np.arange(knum):
            for q3 in np.arange(knum):
                q=[q1,q2,q3]
                Gtau2=convlib.G12_shift(G,q,knum,opt)
                Sig[:,q1,q2,q3]=np.sum(P*Gtau2,axis=(1,2,3))
    return Sig/knum**3

def perturbation(U,beta,knum,taulist,delta):
    eps=lib.calc_disp(knum)
    #PM input
    # G11_tau=PM_G11_tau(beta,eps,taulist)
    # G12_tau=PM_G12_tau(beta,eps,taulist)
    # check(knum,G11_tau) 
    # check(knum,G12_tau)

    #AFM input.
    G11_tau=AFM_G11_tau(beta,eps,taulist,delta)
    G12_tau=AFM_G12_tau(beta,eps,taulist,delta)
    # check(knum,G11_tau)
    # check(knum,G12_tau)
    #-----------1st order: Hartree--------
    G11_iom=convlib.fast_ift_fermion(G11_tau)
    Sigma1_11_iom=np.sum(G11_iom)*U/beta/knum**3*np.ones(N)# Hartree
    #----------2nd order--------
    P11_tau=calc_P(G11_tau,knum,0)
    P12_tau=calc_P(G12_tau,knum,1)
    Sigma2_11_tau=(-1)*U**2/beta**2*calc_Sig(P11_tau,G11_tau,knum,0)
    Sigma2_12_tau=(-1)*U**2/beta**2*calc_Sig(P12_tau,G12_tau,knum,1)
    Sigma2_11_iom=convlib.fast_ift_fermion(Sigma2_11_tau)
    Sigma2_12_iom=convlib.fast_ift_fermion(Sigma2_12_tau)
    Sigma_11_iom=U/2+Sigma1_11_iom[:,None,None,None]+Sigma2_11_iom
    max_sym_index,essential_kpoints, sym_array=lib.calc_sym_array(knum)
    for k in essential_kpoints[:pltkpts]:
        kxind=k[0]
        kyind=k[1]
        kzind=k[2]
        # plt.plot(Sigma_11_iom[n:, kxind, kyind, kzind].real,label='Sigma_11_iom real')
        # plt.plot(Sigma_11_iom[n:, kxind, kyind, kzind].imag,label='Sigma_11_iom imag')
        # plt.grid()
        # plt.legend()
        # plt.title('k=[{},{},{}],eps_k={}'.format(kxind,kyind,kzind,eps[kxind,kyind,kzind]))
        # plt.show()

        # plt.plot(Sigma2_11_iom[n:, kxind, kyind, kzind].real,label='Sigma2_11_iom real')
        # plt.plot(Sigma2_11_iom[n:, kxind, kyind, kzind].imag,label='Sigma2_11_iom imag')
        plt.plot(Sigma2_12_iom[n:, kxind, kyind, kzind].real,label='Sigma2_12_iom real')
        plt.plot(Sigma2_12_iom[n:, kxind, kyind, kzind].imag,label='Sigma2_12_iom imag')
        plt.grid()
        plt.legend()
        plt.title('k=[{},{},{}],eps_k={}'.format(kxind,kyind,kzind,eps[kxind,kyind,kzind]))
        plt.show()
    for i in np.arange(max_sym_index):
        k=essential_kpoints[i]
        kxind=k[0]
        kyind=k[1]
        kzind=k[2]
        if lib.dispersion(kxind/knum,kyind/knum,kzind/knum)**2>0.001:
            plt.scatter(i,Sigma2_12_iom[n,kxind,kyind,kzind].real/lib.dispersion(kxind/knum,kyind/knum,kzind/knum),color='red')
            # plt.scatter(i,sig2_12[n,kxind,kyind,kzind].real/dispersion(kxind/knum,kyind/knum,kzind/knum),color='blue')
            # plt.scatter(i,dispersion(kxind/knum,kyind/knum,kzind/knum)*(sig_new_12[n,0,0,0].real/((-1)*(-6))),color='blue',label='')
    plt.xlabel("different k points")
    plt.title('Sig_12(k)/disp_k at U={},T={}.'.format(U,T))
    plt.show()
    # perturb.FT_test(Sigma2_12_iom,knum,'Sigma2_12_iom',a=1)
    return 0


def perturbation_singlecell(U,beta,knum,taulist):
    #Here we have a different unit cell. Hence new k points.
    klist=2*np.pi*np.arange(knum)/knum
    k1, k2, k3 = np.meshgrid(klist, klist, klist, indexing='ij')
    eps=-2*np.cos(k1)-2*np.cos(k2)-2*np.cos(k3)
    G_tau=singlecell_G11_tau(beta,eps,taulist)
    # G_iom=convlib.fast_ift_fermion(G_tau)
    # in this PM single unit cell case, Hartree is just 0+mu.
    P_tau=calc_P(G_tau,knum,0)
    Sigma2_tau=(-1)*U**2/beta**2*calc_Sig(P_tau,G_tau,knum,0)
    Sigma2_iom=convlib.fast_ift_fermion(Sigma2_tau)
    max_sym_index,essential_kpoints, sym_array=lib.calc_sym_array(knum)
    for k in essential_kpoints[:pltkpts]:
        kxind=k[0]
        kyind=k[1]
        kzind=k[2]
        plt.plot(Sigma2_iom[n:, kxind, kyind, kzind].real,label='Sigma2_iom real')
        plt.plot(Sigma2_iom[n:, kxind, kyind, kzind].imag,label='Sigma2_iom imag')
        plt.grid()
        plt.legend()
        plt.title('k=[{},{},{}],eps_k={}'.format(kxind,kyind,kzind,eps[kxind,kyind,kzind]))
        plt.show()
    Sigma2_NN=np.zeros(N,dtype=complex)
    for kx in np.arange(knum):
        for ky in np.arange(knum):
            for kz in np.arange(knum):
                Sigma2_NN+=Sigma2_iom[:,kx,ky,kz]*np.exp(-1j*kx)/knum**3
    plt.plot(Sigma2_NN[n:].real,label='Sigma2_NN real')
    plt.plot(Sigma2_NN[n:].imag,label='Sigma2_NN imag')
    plt.grid()
    plt.legend()
    plt.show()
    return 0


if __name__ == "__main__":
    #global quantities
    n=500
    N=2*n
    U=0.5
    T=0.01# Note: if use the simplest PM GF without imag part of Sigma, GF can be very ill behaved at low T(T<0.1!)
    beta=1/T
    knum=10
    delta=1
    taulist=(np.arange(N)+0.5)/N*beta
    #plotting control
    pltkpts=20

    #execution
    perturbation(U,beta,knum,taulist,delta)
    # perturbation_singlecell(U,beta,knum,taulist)