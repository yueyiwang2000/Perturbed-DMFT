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
import diagrams
import mpi_module
import copy
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
"""
# Yueyi Wang. May 2024
# This file is a perturbation based on the converged result of DMFT.
This only take the imaginary part of self-energy and manually add a freq-independent splitting to real part as the starting point of perturbation.

"""


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

def iterative_perturbation(om,SigDMFT1,SigDMFT2,U,T,nfreq,order,alpha=1,ifitrative=0,knum=10):
    '''
    the main function doing iterative pertubation. 
    Input:
    SigDMFT1/2: input DMFT self energy.
    U,T: Hubbard U and temperature.
    order: max order taken into account in perturbation. 
    max order number supported:3
    maxit: max number of DMFT self consistant perturbation.
            1: single shot perturbation. 'basic'
            big number: 
    alpha: the factor which suppresses the AFM splitting
    '''

    period=5
    mu=U/2
    beta=1/T
    delta_inf=0
    if ifitrative==1:
        maxit=100 # anyways, a big number which allows self-consistant calculation.
    elif ifitrative==0:
        maxit=1
    # if order==0:
    #     maxit=1
    Sigma11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    Sigma22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    Sigma12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    G11_iom=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    G12_iom=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    G22_iom=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    if rank ==0:
        # print("\t-----DMFT perturbation ----")
        Sigma11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        Sigma11+=ext_sig(SigDMFT1)[:,None,None,None]
        Sigma22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        Sigma22+=ext_sig(SigDMFT2)[:,None,None,None]
        # ori_Sigma11 = ext_sig(SigDMFT1)
        # ori_Sigma22=ext_sig(SigDMFT2)
        

    # this part does not change over iterations.the diagrams can be calculated before iteration.
    # DMFT GF, impurity self-energy: (which is prepared not in multi-process way)
    if rank ==0:
        #--------------Generating Gimp used in counter terms---------------
        iom=np.concatenate((om[::-1],om))*1j
        z_1=z4D(beta,mu,Sigma11,knum,nfreq)#z-delta
        z_2=z4D(beta,mu,Sigma22,knum,nfreq)#z+delta
        G11_iom,G12_iom=G_iterative(knum,z_1,z_2,Sigma12)
        G22_iom=-G11_iom.conjugate()
        G11_tau=fft.fermion_fft_diagG(knum,G11_iom,beta,SigDMFT1,mu)# currently sigma12=0
        G12_tau=fft.fast_ft_fermion(G12_iom,beta)
        G12_tau = np.ascontiguousarray(G12_tau)
        G22_tau=G11_tau[::-1] 
        G22_tau = np.ascontiguousarray(G22_tau)
        G11imp_iom=np.sum(G11_iom,axis=(1,2,3))/knum**3 # impurity GF=sum_k DMFT GF
        G22imp_iom=np.sum(G22_iom,axis=(1,2,3))/knum**3 # impurity GF=sum_k DMFT GF

        #-------------Generating G0=(iom+mu-epsk-Sig_PM-alpha*Sig_AFM)^-1---------
        # here a natural scale of the splitting should be U/2, but that is an upper bound of the splitting.
        # the alpha might be much lower than this value.

        # Sig_PM=(Sigma11+Sigma22)/2
        # Sig_AFM=(Sigma11-Sigma22)/2
        Sig_PM=1j*Sigma11.imag+mu
        Sig_AFM=Sigma11-Sig_PM
        Sigmod11=Sig_PM+alpha*U/2*(1-ifsigAFM)+alpha*Sig_AFM*ifsigAFM
        Sigmod22=Sig_PM-alpha*U/2*(1-ifsigAFM)-alpha*Sig_AFM*ifsigAFM
        Sigmod12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        zmod_1=z4D(beta,mu,Sigmod11,knum,nfreq)#z-delta
        zmod_2=z4D(beta,mu,Sigmod22,knum,nfreq)#z+delta
        G0_11_iom,G0_12_iom=G_iterative(knum,zmod_1,zmod_2,Sigmod12)# currently sigma12=0
        G0_22_iom=-G0_11_iom.conjugate()
        G0_11_tau=fft.fermion_fft_diagG(knum,G0_11_iom,beta,Sigmod11[:,0,0,0],mu)
        G0_12_tau=fft.fast_ft_fermion(G0_12_iom,beta)

        G0_12_tau = np.ascontiguousarray(G0_12_tau)
        G0_22_tau=G0_11_tau[::-1] 
        G0_22_tau = np.ascontiguousarray(G0_22_tau)
        # plt.plot(G0_22_tau[:,0,0,0].real,label='real')
        # plt.plot(G0_22_tau[:,0,0,0].imag,label='imag')
        # plt.legend()
        # plt.show()
        # n0loc11=np.sum(G0_11_iom).real/beta/knum**3+1/2
        # n0loc22=np.sum(G0_22_iom).real/beta/knum**3+1/2
        n0loc11=particlenumber4D(G0_11_iom,beta)
        n0loc22=particlenumber4D(G0_22_iom,beta)
        # print('\tDMFT: n11=',n0loc11,'n22=',n0loc22)

        #-----------counter term skeleton diagrams-------------
        
        delta_inf=np.abs(-mu+SigDMFT1[-1].real)# delta for accurate FFT
        #1st
        # sigimp_1_11=(np.sum(G22imp_iom)/beta+1/2)*U
        # sigimp_1_22=(np.sum(G11imp_iom)/beta+1/2)*U
        sigimp_1_11=(particlenumber1D(G22imp_iom,beta))*U
        sigimp_1_22=(particlenumber1D(G11imp_iom,beta))*U        
        sigimpPM_1=(sigimp_1_11+sigimp_1_22)/2
        sigimpAFM_1=(sigimp_1_11-sigimp_1_22)/2        
        sigimpmod_1_11=sigimpPM_1+alpha*U/2*(1-ifsigAFM)+alpha*sigimpAFM_1*ifsigAFM
        sigimpmod_1_22=sigimpPM_1-alpha*U/2*(1-ifsigAFM)-alpha*sigimpAFM_1*ifsigAFM
        #2nd
        sigimp_2_11,sigimp_2_22=imp.pertimp_func(G11imp_iom,delta_inf,beta,U,knum,2)# 2nd order diagram in Sigma_DMFT
        sigimpPM_2=(sigimp_2_11+sigimp_2_22)/2
        sigimpAFM_2=(sigimp_2_11-sigimp_2_22)/2
        sigimpmod_2_11=sigimpPM_2+alpha*sigimpAFM_2*ifsigAFM
        sigimpmod_2_22=sigimpPM_2-alpha*sigimpAFM_2*ifsigAFM
        #3rd
        sigimp_3_11,sigimp_3_22=imp.pertimp_func(G11imp_iom,delta_inf,beta,U,knum,3)# 3rd order diagram in Sigma_DMFT    
        sigimpPM_3=(sigimp_3_11+sigimp_3_22)/2
        sigimpAFM_3=(sigimp_3_11-sigimp_3_22)/2
        sigimpmod_3_11=sigimpPM_3+alpha*sigimpAFM_3*ifsigAFM
        sigimpmod_3_22=sigimpPM_3-alpha*sigimpAFM_3*ifsigAFM    

    comm.Bcast(G0_11_iom, root=0)
    comm.Bcast(G0_12_iom, root=0)
    comm.Bcast(G0_22_iom, root=0)
    comm.Bcast(G0_11_tau, root=0)
    comm.Bcast(G0_12_tau, root=0)
    comm.Bcast(G0_22_tau, root=0)
    #----------perturbation: skeleton diagrams-----------
    # if order >=1:
    #     n0loc11=np.sum(G0_11_iom).real/beta+1/2
    #     n0loc22=np.sum(G0_22_iom).real/beta+1/2
    if order >=2:
        Sig2_11,Sig2_12=diagrams.sig2(G0_11_tau,G0_12_tau,G0_22_tau,knum,nfreq,U,beta)
        Sig2_22=-Sig2_11.conjugate()

    # for notations, refer to '240126 third order diagram' and also the qualifier paper
    if order>=3:
        Sig3_11,Sig3_12=diagrams.sig3(G0_11_iom,G0_12_iom,G0_11_tau,G0_12_tau,G0_22_tau,knum,nfreq,U,beta)
        Sig3_22=-Sig3_11.conjugate()
        P22_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G0_22_tau,G0_22_tau,0)
        P12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12,G0_12_tau,G0_12_tau,1)



    # ---------------------iteration part: add nonskeletons-------------------
    # In iteration part, we still use dressed GF. But this dressed GF is only for calculation of Tadpoles.
    # Probably this iteration can get accelerated by c like codes?
    it=0
    diff=99999999999
    epsilon=0.00001
    sig_corr_11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    sig_corr_12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    sig_corr_22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    if rank==0:
        diff_arr=np.zeros(maxit)
        m_arr=np.zeros(maxit)
        if order==0:
            m_arr=(n0loc22-n0loc11)*np.ones(maxit)
    if order>0:
        for it in np.arange(maxit):# if no iteration, just go once, if have iteration, 20 iterations.
            if rank ==0:
                # This G_dressed is for the iterated perturbation. But in the 1st iteration is should be G_0.
                z_1=z4D(beta,mu,Sigmod11+sig_corr_11,knum,nfreq)
                z_2=z4D(beta,mu,Sigmod22+sig_corr_22,knum,nfreq)
                Gdress11_iom,Gdress12_iom=G_iterative(knum,z_1,z_2,sig_corr_12)
                Gdress22_iom=-Gdress11_iom.conjugate()
                # nloc11=np.sum(Gdress11_iom).real/knum**3/beta+1/2
                # nloc22=np.sum(Gdress22_iom).real/knum**3/beta+1/2
                nloc11=particlenumber4D(Gdress11_iom,beta)
                nloc22=particlenumber4D(Gdress22_iom,beta)
                m_arr[it]=nloc22-nloc11

                if order >=1:# first order correction to self energy
                    sig_corr1_11=(nloc22*U-sigimpmod_1_11)*np.ones((2*nfreq,knum,knum,knum),dtype=complex)
                    sig_corr1_22=(nloc11*U-sigimpmod_1_22)*np.ones((2*nfreq,knum,knum,knum),dtype=complex)
                    sig_corr_11=copy.deepcopy(sig_corr1_11)
                    sig_corr_22=copy.deepcopy(sig_corr1_22)
                    # print('sig_corr1_11:',nloc22*U-sigimpmod_1_11)
                    sig_corr_12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
                if order >=2:# second order correction to self energy
                    sig_corr2_11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
                    sig_corr2_22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
                    sig_corr2_12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
                    # non-skeleton diagrams(if not doing iterative perturbation) Since we use sig_corr in the first order we have to do non-skeletons first
                    # for second order, it is everything from 1st order inserted in a hartree(which is the only diagram at 1st order)
                    if ifitrative==0:# if alpha not equals to 1 we will need tadpole insertion on tadpole
                        sigext2_11=diagrams.sig2_nonskeleton(G0_22_iom,G0_12_iom,sig_corr_11,sig_corr_22,knum,nfreq,U,beta)
                        sigext2_22=-sigext2_11
                        
                        # print('diagrams order1: {:4f} {:4f}'.format(nloc22*U-sigimpmod_1_11,nloc11*U-sigimpmod_1_22))
                        # print('CT diagrams order2: a={:4f}'.format(sigext2_11.real))
                        sig_corr2_11+=sigext2_11
                        sig_corr2_22+=sigext2_22
                    #skeleton corrections
                    sig_corr2_11+=(Sig2_11-sigimpmod_2_11[:,None,None,None])
                    sig_corr2_22+=(Sig2_22-sigimpmod_2_22[:,None,None,None])
                    sig_corr2_12+=Sig2_12
                    sig_corr_11+=sig_corr2_11
                    sig_corr_22+=sig_corr2_22
                    sig_corr_12+=sig_corr2_12
                if order>=3: # 3rd order correction
                    sig_corr3_11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
                    sig_corr3_22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
                    sig_corr3_12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
                    if ifitrative==0:
                        # non-skeleton diagrams first(if not doing iterative perturbation)
                        #IMPORTANT: iterative perturbation will only cover all insertions in a hartree. It does not cover some other diagrams. e,g, 1st order insertion in a skeleton 2nd order diagram.
                        sigext3a_11=diagrams.sig3_nonskeleton_A(G0_22_iom,G0_12_iom,Sig2_11-sigimpmod_2_11[:,None,None,None],Sig2_22-sigimpmod_2_22[:,None,None,None],Sig2_12,knum,nfreq,U,beta)#second order insertion on a hartree
                        sigext3a_22=-sigext3a_11#
                        # important note: I had a bug for the diagramB and I was using sig_corr1_11. and got ridiculous bugs. BE CAREFUL HERE!
                        #DO NOT USE sig_corr1_11!!!! BUT WHY? because immutable and mutable variables.
                        sigext3b_11=diagrams.sig3_nonskeleton_B(G0_22_iom,G0_12_iom,G0_11_iom,nloc22*U-sigimpmod_1_11,nloc11*U-sigimpmod_1_22,knum,nfreq,U,beta)
                        sigext3b_22=-sigext3b_11
                        sigext3c_11=diagrams.sig2_nonskeleton(G0_22_iom,G0_12_iom,sigext2_11,sigext2_22,knum,nfreq,U,beta)
                        sigext3c_22=-sigext3c_11
                        # print('CT diagrams order3: a={:.4f} b={:.4f} c={:.4f} a+b+c={:.4f}'.format(sigext3a_11.real,sigext3b_11.real,sigext3c_11.real,sigext3a_11.real+sigext3b_11.real+sigext3c_11.real))
                        sig_corr3_11+=(sigext3a_11+sigext3b_11+sigext3c_11)#
                        sig_corr3_22+=(sigext3a_22+sigext3b_22+sigext3c_22)#
                    if it==0:# only calculate once in the 0th iteration, when G=G0, we always need them for ifit =0 or 1.
                        sigext3def_11,sigext3def_12=diagrams.sig3_nonskeleton_DEF(G0_11_iom,G0_12_iom,G0_11_tau,G0_12_tau,P22_tau,P12_tau,nloc22*U-sigimpmod_1_11,nloc11*U-sigimpmod_1_22,beta,knum,nfreq,U)
                        sigext3def_22=-sigext3def_11.conjugate()
                    #skeleton corrections
                    sig_corr3_11+=(Sig3_11+sigext3def_11-sigimpmod_3_11[:,None,None,None])#
                    sig_corr3_22+=(Sig3_22+sigext3def_22-sigimpmod_3_22[:,None,None,None])#
                    sig_corr3_12+=(Sig3_12+sigext3def_12)
                    sig_corr_11+=sig_corr3_11
                    sig_corr_22+=sig_corr3_22
                    sig_corr_12+=sig_corr3_12

            #need a scheme to justify when to converge
            # diff=diff_sigma(Sigma11,new_Sigma11,Sigma12,new_Sigma12)
            # diff_arr[it]=diff
            # if it % period==0:
                # print(f'\tit={it},\tdiff={diff:.7f},\tn11={nloc11:.9f},\tn22={nloc22:.9f}')

            # diff=comm.bcast(diff, root=0)
            # if diff<epsilon:
            #     break
                    
        # finally, according to the sigma, get the best GF.
        sigfinal11=Sigmod11+sig_corr_11
        sigfinal22=Sigmod22+sig_corr_22
        sigfinal12=sig_corr_12
        znew_1=z4D(beta,mu,Sigmod11+sig_corr_11,knum,nfreq)
        znew_2=z4D(beta,mu,Sigmod22+sig_corr_22,knum,nfreq)
        Gdress11_iom,Gdress12_iom=G_iterative(knum,znew_1,znew_2,sig_corr_12)
        Gdress22_iom=-Gdress11_iom.conjugate()
        # nnewloc11=np.sum(Gdress11_iom).real/knum**3/beta+1/2
        # nnewloc22=np.sum(Gdress22_iom).real/knum**3/beta+1/2
        nnewloc11=particlenumber4D(Gdress11_iom,beta)
        nnewloc22=particlenumber4D(Gdress22_iom,beta)
        # Fimp, Eimp,Fdisp,Edisp=energy.PertFreeEnergy(sigfinal11,sigfinal22,sigfinal12,U,T)
    if order==0:
        nnewloc11=n0loc11
        nnewloc22=n0loc22
        # Fimp, Eimp,Fdisp,Edisp=energy.PertFreeEnergy(Sigmod11,Sigmod22,Sigmod12,U,T)
        
    comm.Barrier()

    return nnewloc22-nnewloc11,0,0 #Fdisp,Edisp

def run_perturbation(U,T,nfreq,ordernum,alpha,ifit):
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
        print('reading DMFT data from {}'.format(filename))
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
    mag,Fdisp,Edisp=iterative_perturbation(om,sigA,sigB,U,T,nfreq,ordernum,alpha,ifit)
    return mag,Fdisp,Edisp

def run_all_data():
    '''
    get all energy and magnetization at different U,T,alpha,
    '''
    return 0



if __name__ == "__main__":
    # fileS = 'Sig.OCA'
    # fileD= 'Delta.inp'
    # fileS12='Sig12.dat'
    # some default settings
    knum=10
    nfreq=500
    index=50
    order_arr = np.arange(4)
    U=8.0  
    T=0.05
    if len(sys.argv)>=3:
        U=float(sys.argv[1])
        T=float(sys.argv[2])
    ifit=0# 0: no iteration
    typelist=['basic','iterative']

    # alpha_arr=np.arange(11)/20
    # alpha_arr=np.arange(11)/100
    alpha_arr=np.array(([0.01,0.05,0.1,0.15,0.2,0.3,0.4,0.6,0.8,1.0]))
    magarr=np.zeros((4,alpha_arr.size),dtype=float)
    Earr=np.zeros((4,alpha_arr.size),dtype=float)
    Farr=np.zeros((4,alpha_arr.size),dtype=float)
    print('U={},T={}'.format(U,T))

    ifsigAFM=0# ifsigAFM=1 means use alpha*sigma_AFM, ifsigAFM=0 means a simple splitting alpha*U splitting.
    # ifsigAFM=1 might be better for AFM DMFT solution, but for PM DMFT solution we can only use ifsigAFM=0.

    for order in order_arr:
        for i,alpha in enumerate(alpha_arr):
            print('U={},T={},order={},alpha={}'.format(U,T,order,alpha))
            magarr[order,i],Farr[order,i],Earr[order,i]=run_perturbation(U,T,nfreq,order,alpha,ifit)


    # write the magnetization in the files in perturbation/data
    if rank==0:
        if ifsigAFM==1:
            filename='./magdata/{}_{}_AFMSIG.dat'.format(U,T)
        else:
            filename='./magdata/{}_{}.dat'.format(U,T)
        f = open(filename, 'w')
        for ialp, alp in enumerate(alpha_arr):
            print('{:.2f} {:.5f} {:.5f} {:.5f} {:.5f}'.format(alp,magarr[0,ialp],magarr[1,ialp],magarr[2,ialp],magarr[3,ialp]), file=f)
                          # alpha and magnetization after 0th, 1st 2nd and 3rd order.
        f.close()   
        #also, print the thermodynamics quantities
        if ifsigAFM==1:
            filename='./energydata/{}_{}_AFMSIG.dat'.format(U,T)
        else:
            filename='./energydata/{}_{}.dat'.format(U,T)
        f = open(filename, 'w')
        for ialp, alp in enumerate(alpha_arr):
            print('{:.2f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}'.format(alp,Farr[0,ialp],Farr[1,ialp],Farr[2,ialp],Farr[3,ialp],Earr[0,ialp],Earr[1,ialp],Earr[2,ialp],Earr[3,ialp]), file=f)
                          # alpha and F,E after 0th, 1st 2nd and 3rd order.
        f.close()   