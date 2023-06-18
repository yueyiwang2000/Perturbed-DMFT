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

def FT_3D(knum,G_k,Rx,Ry,Rz,opt=0):# usually R is like (0,0,1)
    n=np.shape(G_k)[0]
    G_real=np.zeros(n,dtype=complex)
    kx,ky,kz=kgrid_full(knum)
    exp_factor=np.exp(-1j*(kx*(Rx+0.5*opt)+ky*Ry+kz*Rz))
    G_real=np.sum(G_k*exp_factor,axis=(1,2,3))/knum**3
    return G_real


def P_offdiag(G,beta):
    n=int(np.shape(G)[0]/4)
    # print(np.shape(G))
    # print('n=',n)
    P_offdiag=np.zeros(2*n+1,dtype=complex)
    for Omind in np.arange(2*n+1):
        P_offdiag[Omind] = np.sum(G[n:3*n] * G[n+Omind-n:3*n+Omind-n])
    return P_offdiag/beta

def Sigma_offdiag(P,G,beta,U):
    n=int(np.shape(G)[0]/4)
    # print(np.shape(G))
    # print('n=',n)
    Sigma_offdiag=np.zeros(2*n,dtype=complex)
    for omind in np.arange(2*n):
        Sigma_offdiag[omind] = np.sum(P* G[n+omind-n:3*n+omind-n+1])
    return -1*Sigma_offdiag*U*U/beta


#--------------test functions below---------

def test(a=1):
    # start_time = time.time()
    #-----------generate G_k--------------
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
    plt.legend()
    plt.show()
    #-------------FT & G_R test--------------
    Gr_p00m=FT_3D(knum,G_off,1,0,0,-1)
    Gr_m00p=FT_3D(knum,G_off,-1,0,0,+1)
    Gr_000m=FT_3D(knum,G_off,0,0,0,-1)
    Gr_000p=FT_3D(knum,G_off,0,0,0,1)
    plt.plot(fermion_om,Gr_p00m.real,label='Grp00m_off_real')
    plt.plot(fermion_om,Gr_p00m.imag,label='Grp00m_off_imag')
    plt.plot(fermion_om,Gr_m00p.real,label='Grm00p_off_real')
    plt.plot(fermion_om,Gr_m00p.imag,label='Grm00p_off_imag')
    plt.plot(fermion_om,Gr_000m.real,label='Gr000m_off_real')
    plt.plot(fermion_om,Gr_000m.imag,label='Gr000m_off_imag')
    plt.plot(fermion_om,Gr_000p.real,label='Gr000p_off_real')
    plt.plot(fermion_om,Gr_000p.imag,label='Gr000p_off_imag')
    plt.legend()
    plt.show()
    #----------P test--------------
    Pr_p00m=P_offdiag(Gr_p00m,beta)
    Pr_m00p=P_offdiag(Gr_m00p,beta)
    Pr_000m=P_offdiag(Gr_000m,beta)
    Pr_000p=P_offdiag(Gr_000p,beta)
    plt.plot(Pr_p00m.real,label='Prp00m_off_real')
    plt.plot(Pr_p00m.imag,label='Prp00m_off_imag')
    plt.plot(Pr_m00p.real,label='Prm00p_off_real')
    plt.plot(Pr_m00p.imag,label='Prm00p_off_imag')
    plt.plot(Pr_000m.real,label='Pr000p_off_real')
    plt.plot(Pr_000m.imag,label='Pr000p_off_imag')
    plt.legend()
    plt.show()
    Sigmar_p00m=Sigma_offdiag(Pr_p00m,Gr_p00m,beta,U)
    Sigmar_m00p=Sigma_offdiag(Pr_m00p,Gr_m00p,beta,U)
    Sigmar_000p=Sigma_offdiag(Pr_000p,Gr_000p,beta,U)
    Sigmar_000m=Sigma_offdiag(Pr_000m,Gr_000m,beta,U)
    plt.plot(Sigmar_p00m.real,label='Sigmarp00m_off_real')
    plt.plot(Sigmar_p00m.imag,label='Sigmarp00m_off_imag')
    plt.plot(Sigmar_m00p.real,label='Sigmarm00p_off_real')
    plt.plot(Sigmar_m00p.imag,label='Sigmarm00p_off_imag')
    plt.plot(Sigmar_000m.real,label='Sigmar000m_off_real')
    plt.plot(Sigmar_000m.imag,label='Sigmar000m_off_imag')
    plt.plot(Sigmar_000p.real,label='Sigmar000p_off_real')
    plt.plot(Sigmar_000p.imag,label='Sigmar000p_off_imag')
    plt.legend()
    plt.show()
    return 0

test()