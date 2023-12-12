import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess,math
import time
sys.path.append('../python_src/')
from mpi4py import MPI
from perturb_lib import *
import perturb_imp as imp
import fft_convolution as fft
import perturb_libmpi as mpilib
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
"""
# Yueyi Wang. Dec 2023
# This file is a perturbation based on the converged result of DMFT.
G^-1=(G^-1_DMFT-Sigma_dynamic-delta_mu-U/2(n^SC-n^0))
This is designed for half-filling. delta_mu=0
This version supports B as a variational parameter, which works as impurity level in DMFT:
G_DMFT=1/(iom+mu-eps_k-B*sigma_z-Sigma_DMFT)
"""



def diff_sigma(sigma11,newsigma11,sigma22,newsigma22):
    res=np.sum(np.abs(sigma11-newsigma11)+np.abs(sigma22-newsigma22))/knum**3
    return res

def iterative_perturbation(SigDMFT1,SigDMFT2,U,T,B,nfreq,order,maxit=250):
    '''
    the main function doiong iterative pertubation. 
    Input:
    SigDMFT1/2: input DMFT self energy.
    U,T: Hubbard U and temperature.
    order: max order taken into account in perturbation. 1 means do nothing, so at least it should be 2.
    max order number supported:3
    maxit: max number of DMFT self consistant perturbation.
    '''
    
    mu=U/2
    beta=1/T
    delta_inf=0
    Sigma11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    Sigma22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    Sigma12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    G11_iom=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    G12_iom=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    P11_tau=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    P12_tau=np.zeros((2*nfreq,knum,knum,knum),dtype=float)
    #quantities with another diagrams in 3rd order.
    if order==3:
        Q11_tau=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        # Q12_tau=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    if rank ==0:
        print("\t-----iterative perturbation (dressed hartree)----")
        time0=time.time()
        Sigma11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        Sigma11+=ext_sig(SigDMFT1)[:,None,None,None]
        Sigma22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        Sigma22+=ext_sig(SigDMFT2)[:,None,None,None]
        ori_Sigma11 = ext_sig(SigDMFT1)
        ori_Sigma22=ext_sig(SigDMFT2)
        

    # this part does not change over iterations. Even the diagrams can be calculated before iteration.
    # DMFT GF, impurity self-energy: (which is prepared not in multi-process way)
    if rank ==0:
        z_1=z4D(beta,mu,Sigma11,knum,nfreq)+B#z-delta
        z_2=z4D(beta,mu,Sigma22,knum,nfreq)-B#z+delta
        G11_iom,G12_iom=G_iterative(knum,z_1,z_2,Sigma12)
        G22_iom=-G11_iom.conjugate()
        # some comments for particle #:
        # to calculate particle numbers we have to get G(\tau=0-) which need GF on imaginary time domain.
        n0loc11=np.sum(G11_iom).real/knum**3/beta+1/2
        n0loc22=np.sum(G22_iom).real/knum**3/beta+1/2
        # print('\told_nloc=',n0loc11)
        G11imp_iom=np.sum(G11_iom,axis=(1,2,3))/knum**3 # impurity GF=sum_k DMFT GF
        delta_inf=np.abs(-mu+SigDMFT1[-1].real)# delta for accurate FFT
        sigimp_2_11,sigimp_2_22=imp.pertimp_func(G11imp_iom,delta_inf,beta,U,knum,2)# 2nd order diagram in Sigma_DMFT
        sigimp_3_11,sigimp_3_22=imp.pertimp_func(G11imp_iom,delta_inf,beta,U,knum,3)# 3rd order diagram in Sigma_DMFT    
    comm.Bcast(G11_iom, root=0)
    comm.Bcast(G12_iom, root=0)
    #----------perturbation-----------
    # For first order, here we do not have to calculate any diagrams.
    # An issue here might be the constant shift for the diagrams. Since we have B so the const shift is different. 
    if order >=2:
    #P_q=\sum_k[G_k(tau)*G_k+q(tau)]*-1/beta
    #sig_tau=np.sum(Pq_tau*Gkq_tau,axis=(1,2,3))*(-1)*U**2/knum**3/beta
        P11_tau=mpilib.precalcP11_mpi(beta,knum,G11_iom,SigDMFT1-B,mu)# here we use Sig_DMFT-B if tricky FFT is needed.
        P12_tau=mpilib.precalcP12_mpi(beta,knum,G12_iom)
        Sig2_11=mpilib.precalcsig_mpi(U,beta,knum,P11_tau,G11_iom,11,SigDMFT1-B,mu)# actually P22 and G11. BUT P11=P22
        Sig2_12=mpilib.precalcsig_mpi_simple(U,beta,knum,P12_tau,G12_iom,12)
        Sig2_22=-Sig2_11.conjugate()


    # order 3 code. should be used after careful review.
    if order>=3:
        # do check those 3rd order diagrams.
        # We do not need Q12. Remember P12=Q12! But we still need Q11, since it is usually not symmetrical for AFM case
        Q11_tau=mpilib.precalcQ_mpi(beta,knum,G11_iom,SigDMFT1,mu,0)  
        # Q12_tau=P12_tau
        #FT
        P11_iom=fft.fast_ift_boson(P11_tau,beta)
        P12_iom=fft.fast_ift_boson(P12_tau,beta)
        Q11_iom=fft.fast_ift_boson(Q11_tau,beta)
        Q12_iom=P12_iom
        #definitions and notations according to qualifier paper. indices are: 111,121,122,112. 

        B_111_tau=fft.precalc_C(P11_iom,P11_iom,beta)
        B_121_tau=fft.precalc_C(P12_iom,P12_iom,beta)
        B_112_tau=2*fft.precalc_C(P11_iom,P12_iom,beta)# This one include 112 and 122 so there is a factor of 2
        A_111_tau=fft.precalc_C(Q11_iom,Q11_iom,beta)
        A_121_tau=fft.precalc_C(Q12_iom,Q12_iom,beta)
        A_112_tau=2*fft.precalc_C(Q11_iom,P12_iom,beta)
      #precalcsig has the factor. (-1)*U**2/knum**3. actually factor needed is U**3. need extra -U.
        Sig3_1_111=-U*mpilib.precalcsigp_mpi(U,beta,knum,A_111_tau,G11_iom,11,SigDMFT1,mu)
        Sig3_1_121=-U*mpilib.precalcsigp_mpi(U,beta,knum,A_121_tau,G11_iom,11,SigDMFT1,mu)
        Sig3_1_112=-U*mpilib.precalcsigp_mpi(U,beta,knum,A_112_tau,G12_iom,12,SigDMFT1,mu)

        Sig3_2_111=-U*mpilib.precalcsig_mpi(U,beta,knum,B_111_tau,G11_iom,11,SigDMFT1,mu)
        Sig3_2_121=-U*mpilib.precalcsig_mpi(U,beta,knum,B_121_tau,G11_iom,11,SigDMFT1,mu)
        Sig3_2_112=-U*mpilib.precalcsig_mpi(U,beta,knum,B_112_tau,G12_iom,12,SigDMFT1,mu)
        Sig3_11=Sig3_1_111+Sig3_2_111+Sig3_1_121+Sig3_2_121
        Sig3_12=Sig3_1_112+Sig3_2_112

    
    # ---------------------iteration part-------------------
    # In iteration part, we still use dressed GF. But this dressed GF is only for calculation of Tadpoles.
    it=0
    diff=99999999999
    epsilon=0#.00001
    if rank==0:
        diff_arr=np.zeros(maxit)
        m_arr=np.zeros(maxit)

    #iteration. this can be done in a single core.
    while it < maxit and diff > epsilon:
        # update the GF which is used for tadpoles.
        if rank ==0:
            z_1=z4D(beta,mu,Sigma11,knum,nfreq)+B#z-delta
            z_2=z4D(beta,mu,Sigma22,knum,nfreq)-B#z+delta
            Gdress11_iom,Gdress12_iom=G_iterative(knum,z_1,z_2,Sigma12)
            Gdress22_iom=-Gdress11_iom.conjugate()
            nloc11=np.sum(Gdress11_iom).real/knum**3/beta+1/2
            nloc22=np.sum(Gdress22_iom).real/knum**3/beta+1/2
            m_arr[it]=nloc22-nloc11
            if order >=1:# first order. Be careful of the sign of B!
                new_Sigma11=ori_Sigma11[:,None,None,None]+B+(nloc22-n0loc22)*U
                new_Sigma22=ori_Sigma22[:,None,None,None]-B+(nloc11-n0loc11)*U
                new_Sigma12=Sigma12
            if order >=2:
                # update: Sigma_new=Sigma(DMFT)+Sigma(pert,first nth order)-Sigma(imp, first nth order)
                new_Sigma11+=Sig2_11-sigimp_2_11[:,None,None,None]
                new_Sigma22+=Sig2_22-sigimp_2_22[:,None,None,None]
                new_Sigma12+=Sig2_12
            if order>=3:
                Sig3_22=-Sig3_11.conjugate()
                new_Sigma11+=Sig3_11-sigimp_3_11[:,None,None,None]
                new_Sigma22+=Sig3_22-sigimp_3_22[:,None,None,None]
                new_Sigma12+=Sig3_12
            timei=time.time()
            diff=diff_sigma(Sigma11,new_Sigma11,Sigma12,new_Sigma12)
            diff_arr[it]=diff
            print(f'\tit={it},\tdiff={diff:.7f},\tn11={nloc11:.9f},\tn22={nloc22:.9f}')
            #second order test
            # plt.plot(Sigma12[nfreq:nfreq+freqdisplayed,0,0,0].real,label='Sigma12 {} real'.format(it))
            # plt.plot(Sigma12[nfreq:nfreq+freqdisplayed,0,0,0].imag,label='Sigma12 {} imag'.format(it))
            # plt.plot(Sigma11[nfreq:nfreq+freqdisplayed,0,0,0].real,label='Sigma11 {} real'.format(it))
            # plt.plot(Sigma11[nfreq:nfreq+freqdisplayed,0,0,0].imag,label='Sigma11 {} imag'.format(it))
            # plt.plot(Sig2_11[nfreq:nfreq+freqdisplayed,0,0,0].real,label='sig_2_11 {} real'.format(it))
            # plt.plot(Sig2_11[nfreq:nfreq+freqdisplayed,0,0,0].imag,label='sig_2_11 {} imag'.format(it))            
            # plt.plot(sigimp_2_11[nfreq:nfreq+freqdisplayed].real,label='sigimp_2_11 {} real'.format(it))
            # plt.plot(sigimp_2_11[nfreq:nfreq+freqdisplayed].imag,label='sigimp_2_11 {} imag'.format(it))
            # 3rd order test
            # plt.plot(Sig3_11[nfreq:nfreq+freqdisplayed,0,0,0].real,label='sig_3_11 {} real'.format(it))
            # plt.plot(Sig3_11[nfreq:nfreq+freqdisplayed,0,0,0].imag,label='sig_3_11 {} imag'.format(it))            
            # plt.plot(sigimp_3_11[nfreq:nfreq+freqdisplayed].real,label='sigimp_3_11 {} real'.format(it))
            # plt.plot(sigimp_3_11[nfreq:nfreq+freqdisplayed].imag,label='sigimp_3_11 {} imag'.format(it))            
            # plt.legend()
            # plt.grid()
            # plt.show()
            Sigma11=new_Sigma11
            Sigma22=new_Sigma22
            Sigma12=new_Sigma12
        
        diff=comm.bcast(diff, root=0)
        comm.Bcast(Sigma11, root=0)
        comm.Bcast(Sigma22, root=0)
        comm.Bcast(Sigma12, root=0)
        it+=1
    if rank==0:
        plt.scatter(diff_arr, m_arr)
        plt.axhline(y=n0loc22-n0loc11, color='r', linestyle='--')
        plt.xlabel("diff")
        plt.ylabel("magnetization")
        plt.show()
    #     plt.plot(ori_Sigma11[nfreq:].real,label='ori_Sigma11 real')
    #     plt.plot(ori_Sigma11[nfreq:].imag,label='ori_Sigma11 imag')
    #     plt.plot(ori_Sigma22[nfreq:].real,label='ori_Sigma22 real')
    #     plt.plot(ori_Sigma22[nfreq:].imag,label='ori_Sigma22 imag')
    #     plt.legend()
    #     plt.show()
    comm.Barrier()
    return Sigma11,Sigma22,Sigma12

if __name__ == "__main__":
    fileS = 'Sig.OCA'
    fileD= 'Delta.inp'
    fileS12='Sig12.dat'
    knum=10 # default
    sig_plot=0  #1=plot 0= do not plot
    pltkpts=1    # max:47 for knum=10
    freqdisplayed=150 # only show first # Matsubara freqs
    ordernum=3# order of perturbation
    if (len(sys.argv)==1):# this is for test
        # standard test
        if rank==0:
            print('This is test mode')
        sig_plot=1# in the test mode, plot the sigma. in the import/calling mode, do not plot.
        U=10.0  
        T=0.43
        B=0.0
        knum=10
        nfreq=500
        index=100
        sigma=np.loadtxt('../files_boldc/{}_{}_{}/Sig.OCA.{}'.format(B,U,T,index))[:nfreq,:]
        # sigma=np.loadtxt('../files_boldc/{}_{}_{}/Sig.out.{}'.format(0,U,T,index))[:nfreq,:]
        # sigma=np.loadtxt('../files_ctqmc/{}_{}/ori_Sig.out.{}'.format(U,T,index))[:nfreq,:]
        sigA=sigma[:,1]+1j*sigma[:,2]#sig+delta
        sigB=U-sigma[:,1]+1j*sigma[:,2]#sig+delta
        if sigma[-1,1]<sigma[-1,3]:
            sigA=sigma[:,3]+1j*sigma[:,2]#sig+delta
            sigB=sigma[:,1]+1j*sigma[:,2]#sig-delta
        iterative_perturbation(sigA,sigB,U,T,B,nfreq,ordernum)
