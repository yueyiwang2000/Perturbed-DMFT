import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess,math
import time
sys.path.append('../python_src/')
from mpi4py import MPI
from perturb_lib import *
import perturb_imp as imp
import fft_convolution as fft
import pert_energy_lib as energy
import perm_def
from scipy.interpolate import interp1d
import diagrams
import mpi_module
import serial_module
import copy
sys.path.append('../python_src/diagramsMC/')
import basis
import svd_diagramsMC_cutPhi
import diag_def_cutPhifast
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
"""
# Yueyi Wang. May 2024
# This file is a perturbation based on the converged result of DMFT.
This only take the imaginary part of self-energy and manually add a freq-independent splitting to real part as the starting point of perturbation.

"""
class params:
    def __init__(self):
        self.Nitt = 5000000   # number of MC steps in a single proc
        self.Ncout = 1000000    # how often to print
        self.Nwarm = 1000     # warmup steps
        self.tmeassure = 10   # how often to meassure
        self.V0norm = 4e-2    # starting V0
        self.recomputew = 5e4/self.tmeassure # how often to check if V0 is correct
        self.per_recompute = 7 # how often to recompute fm auxiliary measuring function

def readDMFT(dir): # read DMFT Sigma and G.
    indexlist=np.arange(200)
    filefound=0
    if (os.path.exists(dir)):
        filename=dir
        filefound=1
    else:
        for i in indexlist[::-1]:
            filename=dir+'.{}'.format(i)
            # print(filename)
            if (os.path.exists(filename)):
                filefound=1
                # print('file found:',filename)
                break
            if i<10:
                # print('warning: only {} DMFT iterations. result might not be accurate!'.format(i))
                break
        

    # if filefound==0:
        # print('{} cannot be found!'.format(filename))  
    # else:
    #     print('reading DMFT data from {}'.format(filename))
    # sigma=np.loadtxt(filename)[:nfreq,:]
    # sigA=sigma[:,1]+1j*sigma[:,2]
    # sigB=sigma[:,3]+1j*sigma[:,4]
    return filename# this also works for G!

def diff_sigma(sigma11,newsigma11,sigma22,newsigma22):
    res=np.sum(np.abs(sigma11-newsigma11)+np.abs(sigma22-newsigma22))/knum**3
    return res

def read_sigimp(U,T):
    filename='./Sigma_imp/coeff_{}_{}.txt'.format(U,T)
    outarray=np.loadtxt(filename,dtype=float)
    # print('shape of outarray',np.shape(outarray))
    filenameu='./Sigma_imp/taubasis.txt'
    ut=np.loadtxt(filenameu).T
    beta=1/T
    lmax=int(outarray[0,0])
    taunum=int(outarray[1,0])
    nfreq=int(outarray[2,0])
    sigimp_1=outarray[3,0]
    cl2=outarray[:lmax,1]
    cl31=outarray[:lmax,2]
    cl32=outarray[:lmax,3]
    cl41=outarray[:lmax,4]
    cl42=outarray[:lmax,5]
    cl43=outarray[:lmax,6]
    cl44=outarray[:lmax,7]
    cl45=outarray[:lmax,8]
    Sigmaimp2=basis.restore_Gf(cl2,ut)
    Sigmaimp31=basis.restore_Gf(cl31,ut)
    Sigmaimp32=basis.restore_Gf(cl32,ut)
    Sigmaimp41=basis.restore_Gf(cl41,ut)
    Sigmaimp42=basis.restore_Gf(cl42,ut)
    Sigmaimp43=basis.restore_Gf(cl43,ut)
    Sigmaimp44=basis.restore_Gf(cl44,ut)
    Sigmaimp45=basis.restore_Gf(cl45,ut)
    # if rank==0:
    #     plt.plot(Sigmaimp44,label='sigimp44')
    #     plt.legend()
    #     plt.show()



    taulist=(np.arange(taunum+1))/taunum*beta
    ori_grid=(np.arange(nfreq*2)+0.5)/(nfreq*2)*beta
    #note: linear interpolation will generate spikes in momentum space. make sure at least use quadratic.
    interpolator_2 = interp1d(taulist, Sigmaimp2, kind='cubic', fill_value='extrapolate')
    interpolator_31 = interp1d(taulist, Sigmaimp31, kind='cubic', fill_value='extrapolate')
    interpolator_32 = interp1d(taulist, Sigmaimp32, kind='cubic', fill_value='extrapolate')
    interpolator_41 = interp1d(taulist, Sigmaimp41, kind='cubic', fill_value='extrapolate')#[1:taunum-1][1:taunum-1]
    interpolator_42 = interp1d(taulist, Sigmaimp42, kind='cubic', fill_value='extrapolate')#[1:taunum-1][1:taunum-1]
    interpolator_43 = interp1d(taulist, Sigmaimp43, kind='cubic', fill_value='extrapolate')#[1:taunum-1][1:taunum-1]
    interpolator_44 = interp1d(taulist, Sigmaimp44, kind='cubic', fill_value='extrapolate')#[1:taunum-1][1:taunum-1]
    interpolator_45 = interp1d(taulist, Sigmaimp45, kind='cubic', fill_value='extrapolate')#[1:taunum-1][1:taunum-1]
    Sigmaimptau2_11=interpolator_2(ori_grid)
    Sigmaimptau31_11=interpolator_31(ori_grid)
    Sigmaimptau32_11=interpolator_32(ori_grid)
    Sigmaimptau41_11=interpolator_41(ori_grid)
    Sigmaimptau42_11=interpolator_42(ori_grid)
    Sigmaimptau43_11=interpolator_43(ori_grid)
    Sigmaimptau44_11=interpolator_44(ori_grid)
    Sigmaimptau45_11=interpolator_45(ori_grid)



    Sigmaimpiom2_11=fft.fermion_ifft(Sigmaimptau2_11,beta)
    Sigmaimpiom31_11=fft.fermion_ifft(Sigmaimptau31_11,beta)
    Sigmaimpiom32_11=fft.fermion_ifft(Sigmaimptau32_11,beta)
    Sigmaimpiom41_11=fft.fermion_ifft(Sigmaimptau41_11,beta)
    Sigmaimpiom42_11=fft.fermion_ifft(Sigmaimptau42_11,beta)
    Sigmaimpiom43_11=fft.fermion_ifft(Sigmaimptau43_11,beta)
    Sigmaimpiom44_11=fft.fermion_ifft(Sigmaimptau44_11,beta)
    Sigmaimpiom45_11=fft.fermion_ifft(Sigmaimptau45_11,beta)
    return ut,lmax,taunum,nfreq,sigimp_1,Sigmaimpiom2_11,Sigmaimpiom31_11,Sigmaimpiom32_11,Sigmaimpiom41_11,Sigmaimpiom42_11,Sigmaimpiom43_11,Sigmaimpiom44_11,Sigmaimpiom45_11,Sigmaimp43,Sigmaimp44

def GetHighFrequency4D(CC,om):
    " Approximates CC ~  A/(i*om-C) "
    A = 1./(1/(CC[-1,:,:,:]*om[-1])).imag
    C = -A*(1./CC[-1,:,:,:]).real# before taking real part this C is purely imaginary.
    return (A, C)


def iterative_perturbation(om,SigDMFT1,SigDMFT2,U,T,nfreq,order,alpha=1):
    '''
    the main function doing iterative pertubation. 
    Input:
    SigDMFT1/2: input DMFT self energy.
    U,T: Hubbard U and temperature.
    order: max order taken into account in perturbation. 
    max order number supported:4
    maxit: max number of DMFT self consistant perturbation.
            1: single shot perturbation. 'basic'
            big number: 
    alpha: the factor which suppresses the AFM splitting
    '''
    # Here we assume there is no iterative perturbation.
    time0=time.time()
    mu=U/2
    beta=1/T
    knum=10
    lmax=1# init. later will read.
    taunum=1# init. later will read.
    nnewloc11=0
    nnewloc22=0
    Sigma11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    Sigma22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    Sigma12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    # Sigmod11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    # Sigmod12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    # Sigmod22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    G0_11_iom=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    G0_12_iom=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    G0_22_iom=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    G0_11_tau=np.zeros((2*nfreq,knum,knum,knum),dtype=float)
    G0_12_tau=np.zeros((2*nfreq,knum,knum,knum),dtype=float)
    G0_22_tau=np.zeros((2*nfreq,knum,knum,knum),dtype=float)
    # if rank ==0:
        # print("\t-----DMFT perturbation ----")
    Sigma11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    Sigma11+=ext_sig(SigDMFT1)[:,None,None,None]
    Sigma22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    Sigma22+=ext_sig(SigDMFT2)[:,None,None,None]
        # ori_Sigma11 = ext_sig(SigDMFT1)
        # ori_Sigma22=ext_sig(SigDMFT2)
        

    # this part does not change over iterations.the diagrams can be calculated before iteration.
    # DMFT GF, impurity self-energy: (which is prepared not in multi-process way)
    
        #--------------Generating Gimp used in counter terms---------------
    # iom=np.concatenate((om[::-1],om))*1j
    # z_1=z4D(beta,mu,Sigma11,knum,nfreq)#z-delta
    # z_2=z4D(beta,mu,Sigma22,knum,nfreq)#z+delta
    # G11_iom,G12_iom=G_iterative(knum,z_1,z_2,Sigma12)
    # G22_iom=-G11_iom.conjugate()
    # G11_tau=fft.fermion_fft_diagG(knum,G11_iom,beta,SigDMFT1,mu)# currently sigma12=0
    # G12_tau=fft.fast_ft_fermion(G12_iom,beta)
    # G12_tau = np.ascontiguousarray(G12_tau)
    # G22_tau=G11_tau[::-1] 
    # G22_tau = np.ascontiguousarray(G22_tau)
    # G11imp_iom=np.sum(G11_iom,axis=(1,2,3))/knum**3 # impurity GF=sum_k DMFT GF
    # G22imp_iom=np.sum(G22_iom,axis=(1,2,3))/knum**3 # impurity GF=sum_k DMFT GF

        #-------------Generating G0=(iom+mu-epsk-Sig_PM-alpha*Sig_AFM)^-1---------
        # here a natural scale of the splitting should be U/2, but that is an upper bound of the splitting.
        # the alpha might be much lower than this value.

        # Sig_PM=(Sigma11+Sigma22)/2
        # Sig_AFM=(Sigma11-Sigma22)/2

    # if rank ==0:    
    #     print('prepare G0...')
    Sig_PM=1j*Sigma11.imag+mu
    Sig_AFM=Sigma11-Sig_PM
    Sigmod11=Sig_PM+alpha*U/2*(1-ifsigAFM)+alpha*Sig_AFM*ifsigAFM
    Sigmod22=Sig_PM-alpha*U/2*(1-ifsigAFM)-alpha*Sig_AFM*ifsigAFM
    Sigmod12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    zmod_1=z4D(beta,mu,Sigmod11,knum,nfreq)#z-delta
    zmod_2=z4D(beta,mu,Sigmod22,knum,nfreq)#z+delta
    G0_11_iom,G0_12_iom=G_iterative(knum,zmod_1,zmod_2,Sigmod12)# currently sigma12=0
    G0_22_iom=-G0_11_iom.conjugate()
    G0_11_tau=fft.fermion_fft_diagG(knum,G0_11_iom,beta,Sigmod11[:,0,0,0],mu).real
    G0_12_tau=fft.fast_ft_fermion(G0_12_iom,beta).real
    G0_11_tau = np.ascontiguousarray(G0_11_tau)
    G0_12_tau = np.ascontiguousarray(G0_12_tau)
    G0_22_tau=G0_11_tau[::-1].real
    G0_22_tau = np.ascontiguousarray(G0_22_tau)
    n0loc11=particlenumber4D(G0_11_iom,beta)
    n0loc22=particlenumber4D(G0_22_iom,beta)

        #-----------counter term skeleton diagrams-------------

    ut,lmax,taunum,nfreq,sigimp_1,Sigmaimpiom2_11,Sigmaimpiom31_11,Sigmaimpiom32_11,Sigmaimpiom41_11,Sigmaimpiom42_11,Sigmaimpiom43_11,Sigmaimpiom44_11,Sigmaimpiom45_11,Sigmaimp43tau,Sigmaimp44tau=read_sigimp(U,T)
    if rank ==0: 
        print('read sigimp... taunum=',taunum)
    Sigmaimpiom3_11=Sigmaimpiom31_11+Sigmaimpiom32_11
    Sigmaimpiom4_11=Sigmaimpiom41_11+Sigmaimpiom42_11+Sigmaimpiom43_11+Sigmaimpiom44_11+Sigmaimpiom45_11
        # now they can be directly used from the previous one.
    #1st
    if sigimp_1>U/2:
        sigimp_1_11=sigimp_1
        sigimp_1_22=U-sigimp_1
    # sigimp_1_11=(particlenumber1D(G22imp_iom,beta))*U# this is the original approach.
    # sigimp_1_22=(particlenumber1D(G11imp_iom,beta))*U    
    else:
        sigimp_1_11=U-sigimp_1
        sigimp_1_22=sigimp_1                
    sigimpPM_1=(sigimp_1_11+sigimp_1_22)/2
    sigimpAFM_1=(sigimp_1_11-sigimp_1_22)/2        
    sigimpmod_1_11=sigimpPM_1+alpha*U/2*(1-ifsigAFM)+alpha*sigimpAFM_1*ifsigAFM
    sigimpmod_1_22=sigimpPM_1-alpha*U/2*(1-ifsigAFM)-alpha*sigimpAFM_1*ifsigAFM
    #2nd
        # sigimp_2_11,sigimp_2_22=imp.pertimp_func(G11imp_iom,delta_inf,beta,U,knum,2)# 2nd order diagram in Sigma_DMFT
    sigimp_2_11=Sigmaimpiom2_11
    sigimp_2_22=-Sigmaimpiom2_11.conjugate()
    sigimpPM_2=(sigimp_2_11+sigimp_2_22)/2
    sigimpAFM_2=(sigimp_2_11-sigimp_2_22)/2
    sigimpmod_2_11=sigimpPM_2+alpha*sigimpAFM_2*ifsigAFM
    sigimpmod_2_22=sigimpPM_2-alpha*sigimpAFM_2*ifsigAFM
    #3rd
    sigimp_3_11=Sigmaimpiom3_11
    sigimp_3_22=-Sigmaimpiom3_11.conjugate()
    # sigimpcheck_3_1_11,sigimpcheck_3_2_11=imp.pertimp_func(G11imp_iom,delta_inf,beta,U,knum,3)# 3rd order diagram in Sigma_DMFT    
    sigimpPM_3=(sigimp_3_11+sigimp_3_22)/2
    sigimpAFM_3=(sigimp_3_11-sigimp_3_22)/2
    sigimpmod_3_11=sigimpPM_3+alpha*sigimpAFM_3*ifsigAFM
    sigimpmod_3_22=sigimpPM_3-alpha*sigimpAFM_3*ifsigAFM    
    #4th
    sigimp_4_11=Sigmaimpiom4_11
    sigimp_4_22=-Sigmaimpiom4_11.conjugate()
    sigimpPM_4=(sigimp_4_11+sigimp_4_22)/2
    sigimpAFM_4=(sigimp_4_11-sigimp_4_22)/2
    sigimpmod_4_11=sigimpPM_4+alpha*sigimpAFM_4*ifsigAFM
    sigimpmod_4_22=sigimpPM_4-alpha*sigimpAFM_4*ifsigAFM   
        # test the sigmaimp read from saved files
    # if rank ==0: 
    #     print('reading sigimp finished! time={}s'.format(time.time()-time0))

        # for testing only. they behaves very nice.
        # print(sigimp_1_11,sigimp_1)
        # plt.plot(sigimp_2_11,label='BF2')
        # plt.plot(Sigmaimpiom2_11,label='restored2')
        # plt.legend()
        # plt.show()

        # plt.plot(sigimp_3_11,label='BF3')
        # plt.plot(Sigmaimpiom3_11,label='restored3')
        # plt.legend()
        # plt.show()

        # plt.plot(Sigmaimpiom4_11,label='restored4')
        # plt.legend()
        # plt.show()
    comm.Barrier()



    # if rank==0:
    #     print('rank{}: lmax before bcast:{}'.format(rank,lmax))
    #     print('rank{}: taunum before bcast:{}'.format(rank,taunum))
    # lmax = comm.bcast(lmax, root=0)
    # taunum = comm.bcast(taunum, root=0)
    # if rank==0:
    #     print('rank{}: lmax after bcast:{}'.format(rank,lmax))
    #     print('rank{}: taunum after bcast:{}'.format(rank,taunum))    
    # # print('rank={},lmax={}'.format(rank,lmax))
    # ut=np.zeros((lmax,taunum+1))
    # if rank==0:
    #     ut=ut_temp
    # print('rank{}: ut before bcast:{}  {}'.format(rank,ut.dtype,ut.shape))
    # comm.Bcast(ut, root=0)
    # print('rank{}: ut after bcast:{}   {}'.format(rank,ut.dtype,ut.shape))



    # comm.Bcast(G0_11_iom, root=0)
    # comm.Bcast(G0_12_iom, root=0)
    # comm.Bcast(G0_22_iom, root=0)
    # comm.Bcast(G0_11_tau, root=0)
    # comm.Bcast(G0_12_tau, root=0)
    # comm.Bcast(G0_22_tau, root=0)



    # ---------------------Perturbation-------------------
    sig_corr_11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    sig_corr_12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    sig_corr_22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    # if rank==0:
    #     for kx in np.arange(knum):
    #         for ky in np.arange(knum):
    #             for kz in np.arange(knum):
    #                 plt.plot(G0_11_tau[:,kx,ky,kz],label='dispersive')
    #                 # plt.plot(Sigmaimpiom44_11,label='imp')
    #                 plt.legend()
    #                 plt.show()
    imax=4
    kbasis=np.empty((imax,knum,knum,knum),dtype=float)
    if rank==0:
        kbasis=basis.gen_kbasis(imax,knum)
    kbasis = np.ascontiguousarray(kbasis)
    comm.Bcast(kbasis, root=0)    
    if order>=1:
        # This G_dressed is for the iterated perturbation. But in the 1st iteration is should be G_0.
        z_1=z4D(beta,mu,Sigmod11+sig_corr_11,knum,nfreq)
        z_2=z4D(beta,mu,Sigmod22+sig_corr_22,knum,nfreq)
        Gdress11_iom,Gdress12_iom=G_iterative(knum,z_1,z_2,sig_corr_12)
        Gdress22_iom=-Gdress11_iom.conjugate()
        # nloc11=np.sum(Gdress11_iom).real/knum**3/beta+1/2
        # nloc22=np.sum(Gdress22_iom).real/knum**3/beta+1/2
        nloc11=particlenumber4D(Gdress11_iom,beta)
        nloc22=particlenumber4D(Gdress22_iom,beta)
        sig_corr1_11=(nloc22*U-sigimpmod_1_11)*np.ones((2*nfreq,knum,knum,knum),dtype=complex)
        sig_corr1_22=(nloc11*U-sigimpmod_1_22)*np.ones((2*nfreq,knum,knum,knum),dtype=complex)
        sig_corr_11=copy.deepcopy(sig_corr1_11)
        sig_corr_22=copy.deepcopy(sig_corr1_22)
        # print('sig_corr1_11:',nloc22*U-sigimpmod_1_11)
        sig_corr_12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        if rank==0:
            print('1st order done! time={}s'.format(time.time()-time0))
    if order>=2:# second order correction to self energy
        time20=time.time()
        # if rank==0:
        #     print('initialization')
        sig_corr2_11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        sig_corr2_22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        sig_corr2_12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        #skeleton corrections: seems better to do serial then broadcast..
        if rank==0:
            P22_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G0_22_tau,G0_22_tau,0)
            P12_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12,G0_12_tau,G0_12_tau,1)

            Sig2_11,Sig2_12=diagrams.sig2(G0_11_tau,G0_12_tau,G0_22_tau,P22_tau,P12_tau,knum,nfreq,U,beta)
            Sig2_22=-Sig2_11.conjugate()
            # print('skeleton')
            time21=time.time()
            # non-skeleton diagrams(if not doing iterative perturbation) Since we use sig_corr in the first order we have to do non-skeletons first
            # for second order, it is everything from 1st order inserted in a hartree(which is the only diagram at 1st order)
            sigext2_11=diagrams.sig2_nonskeleton(G0_22_iom,G0_12_iom,sig_corr1_11,sig_corr1_22,knum,nfreq,U,beta)
            sigext2_22=-sigext2_11
            # print('nonskeleton')
            sig_corr2_11+=(Sig2_11-sigimpmod_2_11[:,None,None,None]+sigext2_11)
            sig_corr2_22+=(Sig2_22-sigimpmod_2_22[:,None,None,None]+sigext2_22)
            sig_corr2_12+=Sig2_12
        # print('bcast')
        comm.Bcast(sig_corr2_11, root=0)
        comm.Bcast(sig_corr2_22, root=0)
        comm.Bcast(sig_corr2_12, root=0)            
        sig_corr_11+=sig_corr2_11
        sig_corr_22+=sig_corr2_22
        sig_corr_12+=sig_corr2_12
        time22=time.time()
        if rank==0:
            # print('2nd order time1={}s  time2={}s'.format(time21-time20,time22-time21))
            print('2nd ordertime={}s'.format(time.time()-time0))
    if order>=3: # 3rd order correction
        time30=time.time()
        sig_corr3_11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        sig_corr3_22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        sig_corr3_12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        if rank==0:
            Q11_tau=serial_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G0_22_tau,G0_11_tau,0)#Q=G_{s',-k}(tau)*G_{s,k+q}(tau)
            Q12_tau=serial_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,12, G0_12_tau,G0_12_tau,1)# Note: G12_-k=-G12_k!
            Q22_tau=serial_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G0_11_tau,G0_22_tau,0)
            Q11_iom=fft.fast_ift_boson(Q11_tau,beta)
            Q22_iom=fft.fast_ift_boson(Q22_tau,beta)
            Q12_iom=fft.fast_ift_boson(Q12_tau,beta)
            R11_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G0_22_tau,G0_11_tau,0)#R=G_{s',k}(-tau)*G_{s,k+q}(tau)
            R12_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12, G0_12_tau,G0_12_tau,1)
            R22_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G0_11_tau,G0_22_tau,0)
            R11_iom=fft.fast_ift_boson(R11_tau,beta)
            R22_iom=fft.fast_ift_boson(R22_tau,beta)
            R12_iom=fft.fast_ift_boson(R12_tau,beta)
            Sig3_1_11,Sig3_1_12,Sig3_2_11,Sig3_2_12=diagrams.sig3(G0_11_iom,G0_12_iom,G0_11_tau,G0_12_tau,G0_22_tau,Q11_iom,Q12_iom,Q22_iom,R11_iom,R12_iom,R22_iom,knum,nfreq,U,beta)
            Sig3_11=Sig3_1_11+Sig3_2_11
            Sig3_12=Sig3_1_12+Sig3_2_12
            Sig3_22=-Sig3_11.conjugate()

            time31=time.time()
                        # non-skeleton diagrams first(if not doing iterative perturbation)
                        #IMPORTANT: iterative perturbation will only cover all insertions in a hartree. It does not cover some other diagrams. e,g, 1st order insertion in a skeleton 2nd order diagram.
            sigext3a_11=diagrams.sig3_nonskeleton_A(G0_22_iom,G0_12_iom,sig_corr2_11,sig_corr2_22,sig_corr2_12,knum,nfreq,U,beta)#second order insertion on a hartree
            sigext3a_22=-sigext3a_11#
                    # important note: I had a bug for the diagramB and I was using sig_corr1_11. and got ridiculous bugs. BE CAREFUL HERE!
                    #DO NOT USE sig_corr1_11!!!! BUT WHY? because immutable and mutable variables. then we have to use deepcopy!!!
            sigext3b_11=diagrams.sig3_nonskeleton_B(G0_22_iom,G0_12_iom,G0_11_iom,sig_corr1_11,sig_corr1_22,knum,nfreq,U,beta)
            sigext3b_22=-sigext3b_11
            # sigext3c_11=diagrams.sig2_nonskeleton(G0_22_iom,G0_12_iom,sigext2_11,sigext2_22,knum,nfreq,U,beta)
            # sigext3c_22=-sigext3c_11
                        # print('CT diagrams order3: a={:.4f} b={:.4f} c={:.4f} a+b+c={:.4f}'.format(sigext3a_11.real,sigext3b_11.real,sigext3c_11.real,sigext3a_11.real+sigext3b_11.real+sigext3c_11.real))
            time32=time.time()
            sigext3def_11,sigext3def_12=diagrams.sig3_nonskeleton_DEF(G0_11_iom,G0_12_iom,G0_11_tau,G0_12_tau,P22_tau,P12_tau,sig_corr1_11,sig_corr1_22,beta,knum,nfreq,U)
            sigext3def_22=-sigext3def_11.conjugate()
        
                    #skeleton corrections
            sig_corr3_11+=(Sig3_11+sigext3def_11-sigimpmod_3_11[:,None,None,None]+sigext3a_11+sigext3b_11)#+sigext3c_11
            sig_corr3_22+=(Sig3_22+sigext3def_22-sigimpmod_3_22[:,None,None,None]+sigext3a_22+sigext3b_22)#+sigext3c_22
            sig_corr3_12+=(Sig3_12+sigext3def_12)


            time33=time.time()
        comm.Bcast(sig_corr3_11, root=0)
        comm.Bcast(sig_corr3_22, root=0)
        comm.Bcast(sig_corr3_12, root=0)            
        sig_corr_11+=sig_corr3_11
        sig_corr_22+=sig_corr3_22
        sig_corr_12+=sig_corr3_12
        if rank==0:
            # print('3nd order timeske={}s  timeabc={}s  timedef={}s'.format(time31-time30,time32-time31,time33-time32))
            print('3rd order done! time={}s'.format(time.time()-time0))
    if order>=4:#Note: here 4th order is only for non-iterative version.
        sig_corr4_11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        sig_corr4_22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        sig_corr4_12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        #summon the monte carlo to get the 2 most difficult diagrams of the 4th order.
        
        GFs=(G0_11_tau.real,G0_12_tau.real,G0_22_tau.real)
        sublatind_basis=np.array([[1,0,0,0],
                              [0,1,0,0],
                              [0,0,1,0],
                              [0,0,0,1]])

        # u is prepared when reading the impurity sigma.
        p=params()

        func43=diag_def_cutPhifast.FuncNDiagNew(T,U,knum,taunum,nfreq,4,ut,kbasis,sublatind_basis,perm_def.perm43,GFs,perm_def.dep43,2)
        func44=diag_def_cutPhifast.FuncNDiagNew(T,U,knum,taunum,nfreq,4,ut,kbasis,sublatind_basis,perm_def.perm44,GFs,perm_def.dep44,4)
        if rank==0:
            print('doing 1st MC..  time={}s'.format(time.time()-time0))        
        Sig4_3_11,Sig4_3_12,Sig4_3_22,Sig4_3_11tau=svd_diagramsMC_cutPhi.Summon_Integrate_Parallel_dispersive(func43,p,imax,lmax,ut,kbasis,sublatind_basis,beta)
        if rank==0:
            print('doing 2nd MC..  time={}s'.format(time.time()-time0))        
        Sig4_4_11,Sig4_4_12,Sig4_4_22,Sig4_4_11tau=svd_diagramsMC_cutPhi.Summon_Integrate_Parallel_dispersive(func44,p,imax,lmax,ut,kbasis,sublatind_basis,beta)
        if rank==0:
            print('MC finished!  time={}s'.format(time.time()-time0))
        time40=time.time()
        # then we evaluate ladder, rpa,....first prepare Q and R. we'll use them later.

        if rank==0:
        # P22_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G0_22_tau,G0_22_tau,0)
        # P12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12,G0_12_tau,G0_12_tau,1)
            P11_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G0_11_tau,G0_11_tau,0)
            P11_iom=fft.fast_ift_boson(P11_tau,beta)
            P22_iom=fft.fast_ift_boson(P22_tau,beta)
            P12_iom=fft.fast_ift_boson(P12_tau,beta)
            # then, generate all diagrams which are 1 insertion in 3rd order. This also have to be done in parallel way.
            Sig4_1_11,Sig4_1_12=diagrams.sig4_1(G0_11_tau,G0_12_tau,G0_22_tau,Q11_iom,Q12_iom,Q22_iom,knum,nfreq,U,beta)
            Sig4_2_11,Sig4_2_12=diagrams.sig4_2(G0_11_tau,G0_12_tau,G0_22_tau,R11_iom,R12_iom,R22_iom,knum,nfreq,U,beta)
            Sig4_5_11,Sig4_5_12=diagrams.sig4_5(G0_11_tau,G0_12_tau,G0_22_tau,P11_iom,P12_iom,P22_iom,knum,nfreq,U,beta)
            # That's all skeleton diagrams of order 4. Then we should consider the 1st order insertion in the 3rd order.

            print('4th order ladderrpa finished!  timeladrpa={}s'.format(time.time()-time0))


            Sig4_3plus_11,Sig4_3plus_12=diagrams.sig4_3plus(G0_11_iom,G0_12_iom,G0_22_iom,G0_11_tau,G0_12_tau,G0_22_tau,  sig_corr1_11,sig_corr1_22,  Q11_iom,Q12_iom,Q22_iom,R11_iom,R12_iom,R22_iom,knum,nfreq,U,beta)
            Sig4_2plus_11,Sig4_2plus_12=diagrams.sig4_2plus(G0_11_iom,G0_12_iom,G0_22_iom,G0_11_tau,G0_12_tau,G0_22_tau,P22_tau,P12_tau,  sig_corr1_11,sig_corr1_22,sig_corr2_11,sig_corr2_12,sig_corr2_22,knum,nfreq,U,beta)
            Sig4_1plus_11=diagrams.sig4_1plus(G0_11_iom,G0_12_iom,G0_22_iom,sig_corr1_11,sig_corr1_22,sig_corr2_11,sig_corr2_12,sig_corr2_22,sig_corr3_11,sig_corr3_12,sig_corr3_22,knum,nfreq,U,beta)

            sig_corr4_11+=(Sig4_1_11+Sig4_2_11+Sig4_3_11+Sig4_4_11+Sig4_5_11-sigimpmod_4_11[:,None,None,None]+Sig4_3plus_11+Sig4_2plus_11+Sig4_1plus_11)#
            sig_corr4_12+=(Sig4_1_12+Sig4_2_12+Sig4_3_12+Sig4_4_12+Sig4_5_12+Sig4_3plus_12+Sig4_2plus_12)  #       
            sig_corr4_22=-sig_corr4_11.conjugate()

            # for kx in np.arange(knum):
            #     for ky in np.arange(knum):
            #         for kz in np.arange(knum):
            #             # plt.plot(Sig4_4_11tau[:,kx,ky,kz],label='dispersive')
            #             # plt.plot(Sigmaimp44tau,label='imp')
            #             # plt.legend()
            #             # plt.show()

            #             # plt.plot(Sig4_1_11[:,kx,ky,kz].real+Sig4_2_11[:,kx,ky,kz].real+Sig4_3_11[:,kx,ky,kz].real+Sig4_4_11[:,kx,ky,kz].real+Sig4_5_11[:,kx,ky,kz].real,label='dispersive real')#
            #             # plt.plot(Sig4_1_11[:,kx,ky,kz].imag+Sig4_2_11[:,kx,ky,kz].imag+Sig4_3_11[:,kx,ky,kz].imag+Sig4_4_11[:,kx,ky,kz].imag+Sig4_5_11[:,kx,ky,kz].imag,label='dispersive imag')#
            #             # # # plt.plot(sigimpmod_41_11.real,label='imp real')
            #             # # # plt.plot(sigimpmod_41_11.imag,label='imp imag')
            #             # plt.plot(sigimpmod_4_11.real,label='imp4 real')
            #             # plt.plot(sigimpmod_4_11.imag,label='imp4 imag')
            #             # plt.legend()
            #             # plt.show()


            #             # plt.plot(Sig4_1_11[:,kx,ky,kz].real,label='dispersive41 real')#+Sig4_2_11[:,kx,ky,kz].real+Sig4_3_11[:,kx,ky,kz].real+Sig4_4_11[:,kx,ky,kz].real+Sig4_5_11[:,kx,ky,kz].real
            #             # plt.plot(Sig4_1_11[:,kx,ky,kz].imag,label='dispersive41 imag')#+Sig4_2_11[:,kx,ky,kz].imag+Sig4_3_11[:,kx,ky,kz].imag+Sig4_4_11[:,kx,ky,kz].imag+Sig4_5_11[:,kx,ky,kz].imag
            #             # plt.plot(Sigmaimpiom41_11.real,label='imp41 real')
            #             # plt.plot(Sigmaimpiom41_11.imag,label='imp41 imag')
            #             # plt.legend()
            #             # plt.show()


            #             # plt.plot(Sig4_2_11[:,kx,ky,kz].real,label='dispersive42 real')
            #             # plt.plot(Sig4_2_11[:,kx,ky,kz].imag,label='dispersive42 imag')
            #             # plt.plot(Sigmaimpiom42_11.real,label='imp42 real')
            #             # plt.plot(Sigmaimpiom42_11.imag,label='imp42 imag')
            #             # plt.legend()
            #             # plt.show()


            #             plt.plot(Sig4_3_11[:,kx,ky,kz].real,label='dispersive43 11 real')
            #             plt.plot(Sig4_3_11[:,kx,ky,kz].imag,label='dispersive43 11 imag')
            #             plt.plot(Sig4_3_22[:,kx,ky,kz].real,label='dispersive43 22 real')
            #             plt.plot(Sig4_3_22[:,kx,ky,kz].imag,label='dispersive43 22 imag')
            #             plt.legend()
            #             plt.show()

            #             # plt.plot(Sigmaimpiom43_11.real,label='imp43 real')
            #             # plt.plot(Sigmaimpiom43_11.imag,label='imp43 imag')
            #             plt.plot(Sig4_3_11tau[:,kx,ky,kz].real,label='dispersive43 11 real')
            #             plt.plot(Sig4_3_11tau[:,kx,ky,kz].imag,label='dispersive43 11 imag')

            #             plt.legend()
            #             plt.show()

            #             plt.plot(Sig4_4_11[:,kx,ky,kz].real,label='dispersive44 11 real')
            #             plt.plot(Sig4_4_11[:,kx,ky,kz].imag,label='dispersive44 11 imag')
            #             plt.plot(Sig4_4_22[:,kx,ky,kz].real,label='dispersive44 22 real')
            #             plt.plot(Sig4_4_22[:,kx,ky,kz].imag,label='dispersive44 22 imag')    
            #             plt.legend()
            #             plt.show()                    
            #             # plt.plot(Sigmaimpiom44_11.real,label='imp44 real')
            #             # plt.plot(Sigmaimpiom44_11.imag,label='imp44 imag')
            #             plt.plot(Sig4_4_11tau[:,kx,ky,kz].real,label='dispersive44 11 real')
            #             plt.plot(Sig4_4_11tau[:,kx,ky,kz].imag,label='dispersive44 11 imag')


            #             plt.legend()
            #             plt.show()


            #             # plt.plot(Sig4_5_11[:,kx,ky,kz].real,label='dispersive45 real')
            #             # plt.plot(Sig4_5_11[:,kx,ky,kz].imag,label='dispersive45 imag')
            #             # plt.plot(Sigmaimpiom45_11.real,label='imp45 real')
            #             # plt.plot(Sigmaimpiom45_11.imag,label='imp45 imag')
            #             # plt.legend()
            #             # plt.show()
            #             # plt.plot(Sig3_1_11[:,kx,ky,kz].real,label='dispersive31 real')
            #             # plt.plot(Sig3_1_11[:,kx,ky,kz].imag,label='dispersive31 imag')
            #             # plt.plot(Sig3_2_11[:,kx,ky,kz].real,label='dispersive32 real')
            #             # plt.plot(Sig3_2_11[:,kx,ky,kz].imag,label='dispersive32 imag')                        
            #             # plt.plot(sigimpmod_3_11.real,label='imp3 real')
            #             # plt.plot(sigimpmod_3_11.imag,label='imp3 imag')
            #             # plt.legend()
            #             # plt.show()
        comm.Bcast(sig_corr4_11, root=0)
        comm.Bcast(sig_corr4_22, root=0)
        comm.Bcast(sig_corr4_12, root=0)                         
        sig_corr_11+=sig_corr4_11
        sig_corr_22+=sig_corr4_22
        sig_corr_12+=sig_corr4_12  
        if rank==0:
            print('4th order done!  time={}s'.format(time.time()-time0))



    # getting the observables
    sigmafilename11='./Sigma_disp/{}_{}/{}_{}_{}_{}_11.dat'.format(U,T,U,T,order,alpha)
    sigmafilename11const='./Sigma_disp/{}_{}/{}_{}_{}_{}_11const.dat'.format(U,T,U,T,order,alpha)
    sigmafilename12='./Sigma_disp/{}_{}/{}_{}_{}_{}_12.dat'.format(U,T,U,T,order,alpha)
    os.makedirs(os.path.dirname(sigmafilename11), exist_ok=True)
    # sigmafilename22='./Sigma_disp/{}_{}_{}_{}_22.dat'.format(U,T,order,alpha)  22 can be got from particle hole symmetry.
    taulist=(np.arange(taunum+1))/taunum*beta
    ori_grid=(np.arange(nfreq*2)+0.5)/(nfreq*2)*beta
    # Then we have to save the self-energy. 
    # However, according to out diagrammatics, any orders of perturbation will contribute at least a constant. which means simple fft does not work.
    if order>0:                
        # finally, according to the sigma, get the best GF.
        sigfinal11=Sigmod11+sig_corr_11
        sigfinal22=Sigmod22+sig_corr_22
        sigfinal12=sig_corr_12
        if rank==0:
            #Note: the sigfinal11 has constant and 1/omega part, which is not suitable for simple fft. 
            # to do a nice fft we have to take care of these things.
            omega=(2*np.arange(2*nfreq)+1-2*nfreq)*np.pi/beta
            sigfinal11_const=sigfinal11[-1,:,:,:].real# This is constant.
            sigfinal11_const_coeffk=basis.coeff_k(sigfinal11_const,kbasis)
            #Note: It's not a good idea to use FFT to treat this const part. Also, svd basis is not good at taking care of these consts.
            # As a result, it's better to create another file to save the consts using kspace basis.
            A,C=GetHighFrequency4D(sigfinal11-sigfinal11_const,omega)
            sigfinal11_om1=A[None,:,:,:]/(1j*omega[:,None,None,None]-C[None,:,:,:])
            sigfinal11_tau1=  A[None,:,:,:]*(fermi(C[None,:,:,:],beta)-1)*np.exp(-ori_grid[:,None,None,None]*C[None,:,:,:])             # analytical FT of A/(iom-C)
            sigfinal11_rest=sigfinal11-sigfinal11_const-sigfinal11_om1
            sigfinaltau11_rest=fft.fast_ft_fermion(sigfinal11_rest,beta).real
            sigfinaltau11=sigfinaltau11_rest+sigfinal11_tau1# Note: This does not include const part!!


            sigfinaltau12=fft.fast_ft_fermion(sigfinal12,beta).real
            # sigfinaltau22=fft.fast_ft_fermion(sigfinal22,beta)
            sigfinaltau11_splined=interp1d(ori_grid, sigfinaltau11, kind='cubic', axis=0,fill_value='extrapolate')
            sigfinaltau12_splined=interp1d(ori_grid, sigfinaltau12, kind='cubic', axis=0,fill_value='extrapolate')
            cli_11=basis.coeff_tk(sigfinaltau11_splined(taulist),ut,kbasis)
            cli_12=basis.coeff_tk(sigfinaltau12_splined(taulist),ut,kbasis)
            # cli_22=basis.coeff_tk(sigfinal22,ut,kbasis)
            np.savetxt(sigmafilename11,cli_11)
            np.savetxt(sigmafilename11const,sigfinal11_const_coeffk)
            np.savetxt(sigmafilename12,cli_12)
            # np.savetxt(sigmafilename22,cli_22)

            # Sig11tk_raw=basis.restore_tk(cli_11,ut,kbasis)
            # Sig12tk_raw=basis.restore_tk(cli_12,ut,kbasis)
            # for kx in np.arange(knum):
            #     for ky in np.arange(knum):
            #         for kz in np.arange(knum):
            #             plt.plot(sigfinal11[:,kx,ky,kz].real-sigfinal11_const[kx,ky,kz].real,label='final11 real')
            #             plt.plot(sigfinal11[:,kx,ky,kz].imag,label='final11 imag')
            #             plt.plot(sigfinal22[:,kx,ky,kz].real,label='final22 real')
            #             plt.plot(sigfinal22[:,kx,ky,kz].imag,label='final22 imag')
            #             plt.legend()
            #             plt.show()
            #             plt.plot(sigfinaltau11[:,kx,ky,kz].real,label='final11 tau')
            #             plt.plot(sigfinaltau12[:,kx,ky,kz].real,label='final12 tau')
            #             plt.legend()
            #             plt.show()
            #             plt.plot(sigfinaltau11_splined(taulist)[:,kx,ky,kz].real,label='final11spline tau')
            #             plt.plot(Sig11tk_raw[:,kx,ky,kz].real,label='final11recovered tau')
            #             plt.plot(sigfinaltau12_splined(taulist)[:,kx,ky,kz].real,label='final12spline tau')
            #             plt.plot(Sig12tk_raw[:,kx,ky,kz].real,label='final12recovered tau')
            #             plt.legend()
            #             plt.show()



        znew_1=z4D(beta,mu,sigfinal11,knum,nfreq)
        znew_2=z4D(beta,mu,sigfinal22,knum,nfreq)
        Gdress11_iom,Gdress12_iom=G_iterative(knum,znew_1,znew_2,sigfinal12)
        Gdress22_iom=-Gdress11_iom.conjugate()
        # nnewloc11=np.sum(Gdress11_iom).real/knum**3/beta+1/2
        # nnewloc22=np.sum(Gdress22_iom).real/knum**3/beta+1/2
        nnewloc11=particlenumber4D(Gdress11_iom,beta)
        nnewloc22=particlenumber4D(Gdress22_iom,beta)
        # Fimp, Eimp,Fdisp,Edisp=energy.PertFreeEnergy(sigfinal11,sigfinal22,sigfinal12,U,T)
    else:
        if rank==0:
            np.savetxt(sigmafilename11,Sigmod11[:,0,0,0])
        # np.savetxt(sigmafilename12,cli_12)# 0th order is diagonal.
        # save sigmod11 and sigmod 22. do not have to use the ut basis.
        nnewloc11=n0loc11
        nnewloc22=n0loc22
        # Fimp, Eimp,Fdisp,Edisp=energy.PertFreeEnergy(Sigmod11,Sigmod22,Sigmod12,U,T)
    
    return nnewloc22-nnewloc11#,Fdisp,Edisp

def run_perturbation(U,T,nfreq,ordernum,alpha):
    if U>=8:# for U>=8 boldc is decent, but below that it is better to choose ctqmc.
        mode='boldc'
    else:
        mode='ctqmc'
    name1='../files_{}/{}_{}/Sig.out'.format(mode,U,T)
    filename1=readDMFT(name1)
    name2='../files_boldc/{}_{}/Sig.OCA'.format(U,T)
    filename2=readDMFT(name2)
    # print(filename1)
    # print(filename2)
    if (os.path.exists(filename1)):
        filename=filename1
    elif (os.path.exists(filename2)):
        filename=filename2
        # print('reading DMFT data from {}'.format(filename))
    else:
        print('{} cannot be found!'.format(filename))  
        return 0  
    sigma=np.loadtxt(filename)[:nfreq,:]
    check=sigma[-1,1]
    om=sigma[:,0]
    # anyways real part of sigA will be greater.
    if check>U/2:
        sigA=sigma[:,1]+1j*sigma[:,2]
        sigB=U-sigma[:,1]+1j*sigma[:,2]
    else:
        sigB=sigma[:,1]+1j*sigma[:,2]
        sigA=U-sigma[:,1]+1j*sigma[:,2]
    mag=iterative_perturbation(om,sigA,sigB,U,T,nfreq,ordernum,alpha)
    #,Fdisp,Edisp
    return mag#,Fdisp,Edisp

def launch_UT(U,T,maxorder):
    ifit=0# 0: no iteration
    typelist=['basic','iterative']
    order_arr = np.arange(maxorder+1)[::-1]
    # alpha_arr=np.arange(11)/20
    # alpha_arr=np.arange(11)/100
    # alpha_arr=np.array(([0.01,0.05,0.1,0.15,0.2,0.3,0.4,0.6,0.8,1.0]))
    alpha_arr=np.array(([0.05,0.1,0.2,0.3,0.6,1.0]))
    # alpha_arr=np.array(([0.0001,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]))
    # alpha_arr=np.array(([1.0,0.05,0.1,0.2,0.3,0.6]))

    magarr=np.zeros((6,alpha_arr.size),dtype=float)
    # Earr=np.zeros((4,alpha_arr.size),dtype=float)
    # Farr=np.zeros((4,alpha_arr.size),dtype=float)
    # print('U={},T={}'.format(U,T))



    for order in order_arr:
        for i,alpha in enumerate(alpha_arr):
            if rank==0:
                print('U={},T={},order={},alpha={} started!'.format(U,T,order,alpha))
            magarr[order,i]=run_perturbation(U,T,nfreq,order,alpha)
            if rank==0:
                print('U={},T={},order={},alpha={} finished!'.format(U,T,order,alpha))            
# ,Farr[order,i],Earr[order,i]

    # write the magnetization in the files in perturbation/data
    if rank==0:
        if ifsigAFM==1:
            filename='./magdata/{}_{}_AFMSIG.dat'.format(U,T)
        else:
            filename='./magdata/{}_{}.dat'.format(U,T)
        f = open(filename, 'w')
        for ialp, alp in enumerate(alpha_arr):
            print('{:.2f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}'.format(alp,magarr[0,ialp],magarr[1,ialp],magarr[2,ialp],magarr[3,ialp],magarr[4,ialp]), file=f)
                          # alpha and magnetization after 0th, 1st 2nd and 3rd order.
        f.close()  



        #also, print the thermodynamics quantities
        # if ifsigAFM==1:
        #     filename='./energydata/{}_{}_AFMSIG.dat'.format(U,T)
        # else:
        #     filename='./energydata/{}_{}.dat'.format(U,T)
        # f = open(filename, 'w')
        # for ialp, alp in enumerate(alpha_arr):
        #     print('{:.2f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}'.format(alp,Farr[0,ialp],Farr[1,ialp],Farr[2,ialp],Farr[3,ialp],Earr[0,ialp],Earr[1,ialp],Earr[2,ialp],Earr[3,ialp]), file=f)
        #                   # alpha and F,E after 0th, 1st 2nd and 3rd order.
        # f.close()   


def run_pert():
    # T_bound=np.array(((3.0,0.08,0.14),(5.,0.2,0.31),
    #                   (8.,0.2,0.4),(10.,0.25,0.5),(12.,0.24,0.4),(14.,0.26,0.4)))
    T_bound=np.array(((3.0,0.08,0.14),(5.,0.2,0.31),
                      (8.,0.4,0.59),(10.,0.5,0.63),(12.,0.4,0.6),(14.,0.4,0.5)))
    #(4.,0.1,0.25),(6.,0.27,0.37),(7.,0.27,0.4),(9.,0.28,0.45),(11.,0.3,0.5),
    for list in T_bound:
        U=list[0]
        # print(U)
        
        for T in np.arange(int(list[1]*100),int(list[2]*100))/100:
            filename='./Sigma_disp/{}_{}/'.format(U,T)
            if (os.path.exists(filename))==0:
                if rank==0:
                    print(U,T)
                launch_UT(U,T,4)
            else:
                if rank==0:
                    print('skip U={} T={}'.format(U,T))

if __name__ == "__main__":
    # fileS = 'Sig.OCA'
    # fileD= 'Delta.inp'
    # fileS12='Sig12.dat'
    # some default settings

    ifsigAFM=0# ifsigAFM=1 means use alpha*sigma_AFM, ifsigAFM=0 means a simple splitting alpha*U splitting.
    # ifsigAFM=1 might be better for AFM DMFT solution, but for PM DMFT solution we can only use ifsigAFM=0.

    knum=10
    nfreq=500
    index=50
    
    U=10.0  
    T=0.35
    if len(sys.argv)>=3:
        U=float(sys.argv[1])
        T=float(sys.argv[2])
    # Tlist=np.array([0.05,0.25,0.35,0.4,0.45])
    # for T in Tlist:
    # launch_UT(10.,0.29,4)
    # launch_UT(3.,0.08,4)
    # launch_UT(3.,0.1,4)
    # launch_UT(3.,0.12,4)
    # launch_UT(3.,0.13,4)
    # launch_UT(5.,0.2,4)
    # launch_UT(5.,0.28,4)
    # launch_UT(8.,0.33,4)
    # launch_UT(8.,0.36,4)
    # launch_UT(10.,0.36,4)
    # launch_UT(12.,0.26,4)
    # launch_UT(12.,0.36,4)
    # launch_UT(14.,0.26,4)
    # launch_UT(14.,0.36,4)
    # launch_UT(10.,0.35,4)
    run_pert()