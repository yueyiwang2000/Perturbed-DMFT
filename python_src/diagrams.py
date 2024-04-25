import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess,math
import time
sys.path.append('../python_src/')
from mpi4py import MPI
from perturb_lib import *
import perturb_imp as imp
import fft_convolution as fft
import mpi_module
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

'''
This file aims to pack the algorithm of diagrams. Our project sometimes requires different propagators in a diagrams.
'''





def sig1(G11_iom,G22_iom,knum,U,beta): # calculate all 1st order diagrams: 1st order, 1 fermion loop, ==> plus sign
    Sig1_11=(-1)*(-1)*U*(np.sum(G22_iom).real/knum**3/beta+1/2)# only works for half filling.
    Sig1_22=(-1)*(-1)*U*(np.sum(G11_iom).real/knum**3/beta+1/2)
    return Sig1_11,Sig1_22

def sig2(G11_tau,G12_tau,G22_tau,knum,nfreq,U,beta): # calculate all 2nd order diagrams
    '''
    This function packs all 2nd order diagrams. and the result is in freq space.
    '''
    # G11_tau=fft.fermion_fft_diagG(knum,G11_iom,beta,SigDMFT1-B,mu)
    # G12_tau=fft.fast_ft_fermion(G12_iom,beta)
    P22_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G22_tau,G22_tau,0)
    P12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G12_tau,G12_tau,1)
    Sig2_11=mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11, G11_tau,P22_tau,beta,U,0)
    Sig2_12=mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12, G12_tau,P12_tau,beta,U,1)
    # Sig2_22=-Sig2_11.conjugate()
    # Sig2tau_11=mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,11, G11_tau,P22_tau,beta,U,0)
    return Sig2_11,Sig2_12#,Sig2tau_11


def sig2_nonskeleton(G22_iom,G12_iom,sig1_11,sig1_22,knum,nfreq,U,beta):
    '''
    this function calculate non-skeleton diagrams of 2rd order.
    Currently this contain a tadpole insertion on a tadpole.
    '''
    GSigG22=G22_iom*sig1_22*G22_iom+G12_iom*sig1_11*G12_iom
    # plt.plot(GSigG22[:,0,0,0].real,label='real')
    # plt.plot(GSigG22[:,0,0,0].imag,label='imag')
    # plt.legend()
    # plt.show()
    sigext11=np.sum(GSigG22)*U/knum**3/beta
    return sigext11

def sig3(G11_iom,G12_iom,G11_tau,G12_tau,G22_tau,knum,nfreq,U,beta): # calculate all 3rd order diagrams
    '''
    This function packs all 3rd order diagrams. and the result is in freq space.
    '''
    # do check those 3rd order diagrams.
    Q11_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G22_tau,G11_tau,0)#Q=G_{s',-k}(tau)*G_{s,k+q}(tau)
    Q12_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, -G12_tau,G12_tau,1)# Note: G12_-k=-G12_k!
    Q22_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G11_tau,G22_tau,0)
    R11_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G22_tau,G11_tau,0)#R=G_{s',k}(-tau)*G_{s,k+q}(tau)
    R12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G12_tau,G12_tau,1)
    R22_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G11_tau,G22_tau,0)
    # P22_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G22_tau,G22_tau,0)
    # P12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G12_tau,G12_tau,1)
    # Note1: Polarization P contains 2 propagators with same spin. But this is not the case for 3rd order.
    # Note2: Q11,Q12,R11,R12 are all symmetric in k space. see proof in '240126 third order diagram'
    #FT
    R11_iom=fft.fast_ift_boson(R11_tau,beta)
    R22_iom=fft.fast_ift_boson(R22_tau,beta)
    R12_iom=fft.fast_ift_boson(R12_tau,beta)
    Q11_iom=fft.fast_ift_boson(Q11_tau,beta)
    Q22_iom=fft.fast_ift_boson(Q22_tau,beta)
    Q12_iom=fft.fast_ift_boson(Q12_tau,beta)
    #definitions and notations according to qualifier paper. indices are: 111,121,122,112. 

    B_111_tau=fft.precalc_C(R11_iom,R11_iom,beta)
    B_121_tau=fft.precalc_C(R12_iom,R12_iom,beta)
    B_112_tau=fft.precalc_C(R11_iom,R12_iom,beta)
    B_122_tau=fft.precalc_C(R12_iom,R22_iom,beta)
    A_111_tau=fft.precalc_C(Q11_iom,Q11_iom,beta)
    A_121_tau=fft.precalc_C(Q12_iom,Q12_iom,beta)
    A_112_tau=fft.precalc_C(Q11_iom,Q12_iom,beta)
    A_122_tau=fft.precalc_C(Q12_iom,Q22_iom,beta)
    #precalcsig has the factor. (-1)*U**2/knum**3. actually factor needed is U**3. need extra -U.
    # Note: calculations below are simplified using symmetries of k, tau, and spin. for details, see '240126 third order diagram'.
    Sig3_1_111=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,-G11_tau,A_111_tau,beta,U,0 )
    Sig3_1_121=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,-G11_tau,A_121_tau,beta,U,0 )
    Sig3_1_112=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,A_112_tau,beta,U,1 )# check here.
    Sig3_1_122=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,A_122_tau,beta,U,1 )

    Sig3_2_111=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,G22_tau,B_111_tau,beta,U,0 )
    Sig3_2_121=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,G22_tau,B_121_tau,beta,U,0 )
    Sig3_2_112=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,B_112_tau,beta,U,1 )
    Sig3_2_122=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,B_122_tau,beta,U,1 )



    Sig3_11=Sig3_2_111+Sig3_1_111+Sig3_1_121+Sig3_2_121
    Sig3_12=Sig3_1_112+Sig3_2_112+Sig3_1_122+Sig3_2_122

    return Sig3_11,Sig3_12

def sig3_nonskeleton_A(G22_iom,G12_iom,sig2iom_11,sig2iom_22,sig2iom_12,knum,nfreq,U,beta):
    '''
    this function calculate non-skeleton diagrams of 3rd order.
    Currently this contain a second order insertion on a tadpole.
    '''
    GSigG22=G22_iom*sig2iom_22*G22_iom+G12_iom*sig2iom_12*G22_iom*2+G12_iom*sig2iom_11*G12_iom
    sigext11=np.sum(GSigG22)*U/knum**3/beta
    return sigext11

def sig3_nonskeleton_B(G22_iom,G12_iom,G11_iom,sig1_11,sig1_22,knum,nfreq,U,beta):
    '''
    this diagram is 2 hartree insertion on a hartree so it is 3rd order
    '''
    GSigGSigG22=G22_iom**3*sig1_22**2 + G12_iom**2*G11_iom*sig1_11**2 + G12_iom**2*G22_iom*sig1_11*sig1_22*2
    sigext11=np.sum(GSigGSigG22)*U/knum**3/beta
    return sigext11 


def sig3_nonskeleton_DEF(G11_iom,G12_iom,G11_tau,G12_tau,P22_tau,P12_tau,sig1_11,sig1_22,beta,knum,nfreq,U):
    '''
    3 third order diagrams, which are 1 B insertion in second order diagrams.
    In this function, we treat 1 B insertion as order 1. This is kleinert's idea.
    '''
    G22_iom=-G11_iom.conjugate()
    G22_tau=G11_tau[::-1]
    if rank ==0:
        # note: spin indecies are all up. 22= spin dn.
        GsigG11_iom=G11_iom**2*sig1_11+G12_iom**2*sig1_22
        GsigG12_iom=G12_iom*G22_iom*sig1_22+G11_iom*G12_iom*sig1_11
        GsigG22_iom=G22_iom**2*sig1_22+G12_iom**2*sig1_11
        GsigG11_tau=fft.fast_ft_fermion(GsigG11_iom,beta)# GBG scales at least as 1/omega**2.
        GsigG12_tau=fft.fast_ft_fermion(GsigG12_iom,beta)
        GsigG22_tau=fft.fast_ft_fermion(GsigG22_iom,beta)
    # should i broadcast?
    GsigG11_tau=np.ascontiguousarray(GsigG11_tau)
    GsigG12_tau=np.ascontiguousarray(GsigG12_tau)
    GsigG22_tau=np.ascontiguousarray(GsigG22_tau)
    comm.Bcast(GsigG11_tau, root=0)
    comm.Bcast(GsigG12_tau, root=0)
    comm.Bcast(GsigG22_tau, root=0)
    Pa22_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G22_tau,GsigG22_tau,0)
    Pa12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G12_tau,GsigG12_tau,1)
    Pb22_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,GsigG22_tau,G22_tau,0)
    Pb12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,GsigG12_tau,G12_tau,1)
    # siga-c: sig1 insertion in a second order diagram. 
    Sigd11=mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11, G11_tau,Pa22_tau,beta,U,0)
    Sigd12=mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12, G12_tau,Pa12_tau,beta,U,1)
    Sige11=mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11, G11_tau,Pb22_tau,beta,U,0)
    Sige12=mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12, G12_tau,Pb12_tau,beta,U,1)    
    Sigf11=mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11, GsigG11_tau,P22_tau,beta,U,0)
    Sigf12=mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12, GsigG12_tau,P12_tau,beta,U,1)       
    Sig11=Sigd11+Sige11+Sigf11
    Sig12=Sigd12+Sige12+Sigf12
    return Sig11,Sig12


