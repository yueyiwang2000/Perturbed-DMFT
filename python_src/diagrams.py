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

def insertion_Bequ1_order3(G11_iom,G12_iom,G11_tau,G12_tau,P11_tau,P12_tau,beta,B,knum,nfreq,U):
    '''
    3 third order diagrams, which are 1 B insertion in second order diagrams.
    In this function, we treat 1 B insertion as order 1. This is kleinert's idea.
    '''
    G22_iom=-G11_iom.conjugate()
    if rank ==0:
        GBG11_iom=(G11_iom**2+G12_iom**2)*B
        GBG12_iom=(G12_iom*G22_iom+G11_iom*G11_iom)*B
        GBG11_tau=fft.fast_ft_fermion(GBG11_iom,beta)# GBG scales at least as 1/omega**2.
        GBG12_tau=fft.fast_ft_fermion(GBG12_iom,beta)
    Pa11_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G11_tau,GBG11_tau,0)
    Pa12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12,G12_tau,GBG12_tau,1)
    Pb11_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,GBG11_tau,G11_tau,0)
    Pb12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12,GBG12_tau,G12_tau,1)
    Siga11=mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11, G11_tau,Pa11_tau,beta,U,0)
    Siga12=mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12, G12_tau,Pa12_tau,beta,U,1)
    Sigb11=mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11, G11_tau,Pb11_tau,beta,U,0)
    Sigb12=mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12, G12_tau,Pb12_tau,beta,U,1)    
    Sigc11=mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11, GBG11_tau,P11_tau,beta,U,0)
    Sigc12=mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12, GBG12_tau,P12_tau,beta,U,1)       
    Sig11=Siga11+Sigb11+Sigc11
    Sig12=Siga12+Sigb12+Sigc12
    return Sig11,Sig12

def insertion_Bequ0_order3(G11_iom,G12_iom,G11_tau,G12_tau,P11_tau,P12_tau,beta,delta_n1,delta_n2,knum,nfreq,U):
    '''
    3 third order diagrams, which are 1 B insertion in second order diagrams.
    In this function, we treat any # of B insertion as order 0. 
    This might be reasonable because the insertion of B does not have anything with U term, which means the order or xi is unchanged.
    But this will finally effectively lead to something like B=0 in perturbation?
    '''

    G22_iom=-G11_iom.conjugate()
    if rank ==0:
        GBG11_iom=G11_iom**2*delta_n2*U+G12_iom**2*delta_n1*U
        GBG12_iom=G12_iom*G22_iom*delta_n1*U+G11_iom*G11_iom*delta_n2*U
        GBG11_tau=fft.fast_ft_fermion(GBG11_iom,beta)# GBG scales at least as 1/omega**2.
        GBG12_tau=fft.fast_ft_fermion(GBG12_iom,beta)
    Pa11_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G11_tau,GBG11_tau,0)
    Pa12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12,G12_tau,GBG12_tau,1)
    Pb11_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,GBG11_tau,G11_tau,0)
    Pb12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12,GBG12_tau,G12_tau,1)
    Siga11=mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11, G11_tau,Pa11_tau,beta,U,0)
    Siga12=mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12, G12_tau,Pa12_tau,beta,U,1)
    Sigb11=mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11, G11_tau,Pb11_tau,beta,U,0)
    Sigb12=mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12, G12_tau,Pb12_tau,beta,U,1)    
    Sigc11=mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11, GBG11_tau,P11_tau,beta,U,0)
    Sigc12=mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12, GBG12_tau,P12_tau,beta,U,1)       
    Sig11=Siga11+Sigb11+Sigc11
    Sig12=Siga12+Sigb12+Sigc12
    return Sig11,Sig12