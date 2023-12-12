import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess,math
import time
from mpi4py import MPI
from perturb_lib import *
import perturb_imp as imp
import fft_convolution as fft
import perturb_libmpi as mpilib
from memory_profiler import profile
import psutil
import copy
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
"""
# Yueyi Wang. Sept 2023
# This file is usually not called or imported in other python files. Just for test and use command line to run it.
sc_pert=self-consistent perturbation 
"""
def output_memory_usage():
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)  
    # print(f"Memory usage: {memory_usage:.2f} MB")
    return memory_usage


def diff_sigma(sigma11,newsigma11,sigma22,newsigma22):
    res=np.sum(np.abs(sigma11-newsigma11)+np.abs(sigma22-newsigma22))/knum**3
    return res

# @profile
def func_diffhartree(extramu,z01,z02,hartree,knum):
    k1,k2,k3=gen_full_kgrids(knum)
    disp=dispersion(k1,k2,k3)
    # print(np.shape(z01))
    G_iom = (z02+extramu)[:,None,None,None]/ ((z01-extramu)[:,None,None,None]*(z02+extramu)[:,None,None,None] - disp**2)
    # G_iom=G_11(knum,z01-extramu,z02+extramu)
    return np.sum(G_iom/knum**3).real-hartree
    
# @profile
def shifted_Gloc(hartree,oriSigma11,oriSigma22,delta,beta,mu,knum):
    '''
    This function is used to shift a muhat*sigma_z in G=(iom+mu+muhat*sigmaz-epsk-sigma_loc)to make sure after perturbation the particle number is conserved.
    '''
    z_1=z(beta,mu,oriSigma11,nfreq)#z-delta
    z_2=z(beta,mu,oriSigma22,nfreq)#z+delta
    extramu_lower=-0.9*delta
    extramu_upper=3
    mu_epsilon=1e-6*delta# should be proportional to delta.
    diff_upper=func_diffhartree(extramu_upper,z_1,z_2,hartree,knum)
    diff_lower=func_diffhartree(extramu_lower,z_1,z_2,hartree,knum)
    if diff_upper*diff_lower>0:
        print('diff_upper*diff_lower>0!')
        return 0,0
    while extramu_upper-extramu_lower>mu_epsilon:
        extramu_middle=(extramu_upper+extramu_lower)/2
        diff_middle=func_diffhartree(extramu_middle,z_1,z_2,hartree,knum)
        if diff_upper*diff_middle>0:
            extramu_upper=copy.deepcopy(extramu_middle)
        else:
            extramu_lower=copy.deepcopy(extramu_middle)
    extramu=(extramu_upper+extramu_lower)/2
    print('\tshifted mu:',extramu/delta,'delta')
    eff_delta=delta+extramu
    k1,k2,k3=gen_full_kgrids(knum)
    disp=dispersion(k1,k2,k3)[None,:,:,:]
    Gnewiom = (z_2+extramu)[:,None,None,None]/ ((z_1-extramu)[:,None,None,None]*(z_2+extramu)[:,None,None,None] - disp**2)
    Gloc=np.sum(Gnewiom,axis=(1,2,3))/knum**3
    return eff_delta,Gloc


def iterative_perturbation(SigDMFT1,SigDMFT2,U,T,nfreq,order,maxit=5):
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
    delta_eff=0
    Sigma11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    Sigma22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    Sigma12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)#after 1st iteration we have off diagonal self-energy! this should also be updated.
    G11_iom=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    G12_iom=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    P11_tau=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    P12_tau=np.zeros((2*nfreq,knum,knum,knum),dtype=float)
    #quantities with another diagrams in 3rd order.
    if order==3:
        Q11_tau=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        # Q12_tau=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    if rank ==0:
        print("\t----------iterative perturbation---------")
        time0=time.time()
        Sigma11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        Sigma11+=ext_sig(SigDMFT1)[:,None,None,None]
        Sigma22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        Sigma22+=ext_sig(SigDMFT2)[:,None,None,None]
        ori_Sigma11 = ext_sig(SigDMFT1)
        ori_Sigma22=ext_sig(SigDMFT2)

    #----------------double counted part-----------
    # this part does not change over iterations.
    # impurity GF:
    if rank ==0:
        z_1=z4D(beta,mu,Sigma11,knum,nfreq)#z-delta
        z_2=z4D(beta,mu,Sigma22,knum,nfreq)#z+delta
        G11ori_iom=G_11(knum,z_1,z_2)
        oldHartreeloc=np.sum(G11ori_iom).real/knum**3
        print('\told_nloc=',oldHartreeloc/beta)
        G0_11iom=G_11(knum,z_1,z_2) 
        G11imp_iom=np.sum(G11ori_iom,axis=(1,2,3))/knum**3
        # G22imp_iom=-G11imp_iom.conjugate()
        delta_inf=np.abs(-mu+SigDMFT1[-1].real)
        sigimp11,sigimp22=imp.pertimp_func(G11imp_iom,delta_inf,beta,U,knum,order)

    
    # ---------------------iteration part-------------------
    #preparation
    it=0
    diff=99999999999
    epsilon=.00001


    #iteration
    while it < maxit and diff > epsilon:
        if rank ==0:
            z_1=z4D(beta,mu,Sigma11,knum,nfreq)#z-delta
            z_2=z4D(beta,mu,Sigma22,knum,nfreq)#z+delta
            G11_iom,G12_iom=G_iterative(knum,z_1,z_2,Sigma12)
            Hartree11=np.sum(G11_iom).real/knum**3
           
            delta_eff,G11_loc=shifted_Gloc(Hartree11,ori_Sigma11,ori_Sigma22,delta_inf,beta,mu,knum)
            # print('delta_eff=',delta_eff)
            Hartreeloc=np.sum(G11_loc).real
            # print('\tn11=',Hartree11/beta,'new_nloc=',Hartreeloc/beta)
            # print(np.shape(G11_loc))
            sigimp11,sigimp22=imp.pertimp_func(G11_loc,delta_inf,beta,U,knum,order)#G11_loc or G11imp_iom?
        # Sig1_11=U/beta*np.sum(G11_iom)/knum**3*np.ones_like(G11_iom)
        # Sig1_22=-Sig1_11
        comm.Bcast(G11_iom, root=0)
        comm.Bcast(G12_iom, root=0)
        
        P11_tau=mpilib.precalcP11_mpi(beta,knum,G11_iom,SigDMFT1,mu)  
        P12_tau=mpilib.precalcP12_mpi(beta,knum,G12_iom)
        Sig2_11=mpilib.precalcsig_mpi(U,beta,knum,P11_tau,G11_iom,11,SigDMFT1,mu)# actually P22 and G11. BUT P11=P22
        Sig2_12=mpilib.precalcsig_mpi_simple(U,beta,knum,P12_tau,G12_iom,12)
        if order==3:
            # We do not need Q12. Remember P12=Q12! But we still need Q11, since it is usually not symmetrical for AFM case
            Q11_tau=mpilib.precalcQ_mpi(beta,knum,G11_iom,SigDMFT1,mu,0)  
            P11_iom=fft.fast_ift_boson(P11_tau)
            P12_iom=fft.fast_ift_boson(P12_tau)
            Q11_iom=fft.fast_ift_boson(Q11_tau)
            C1_111_tau=(-1)*U*fft.precalc_C(P11_iom,P11_iom)/beta
            C1_121_tau=(-1)*U*fft.precalc_C(P12_iom,P12_iom)/beta# hence this one only uses P12, and Q12=P12, so C1_121=C2_121!
            C1_112_tau=2*(-1)*U*fft.precalc_C(P11_iom,P12_iom)/beta# This one include 112 and 121 so there is a factor of 2
            C2_111_tau=(-1)*U*fft.precalc_C(Q11_iom,Q11_iom)/beta
            C2_112_tau=2*(-1)*U*fft.precalc_C(Q11_iom,P12_iom)/beta
            #Note:for P we have coeff 1/beta. C has 1/beta**2. sig has -1*U**2/beta**3. what we should have is U**3/beta**3. so we need a extra -U.
            #sig3_1_111 means: 3rd order, 1st diagram, tau indices are 111.
            Sig3_1_111=mpilib.precalcsig_mpi(U,beta,knum,C1_111_tau,G11_iom,11,SigDMFT1,mu)
            Sig3_1_121=mpilib.precalcsig_mpi(U,beta,knum,C1_121_tau,G11_iom,11,SigDMFT1,mu)
            Sig3_1_112=mpilib.precalcsig_mpi(U,beta,knum,C1_112_tau,G12_iom,12,SigDMFT1,mu)
            Sig3_2_111=mpilib.precalcsigp_mpi(U,beta,knum,C2_111_tau,G11_iom,11,SigDMFT1,mu)
            Sig3_2_121=mpilib.precalcsigp_mpi(U,beta,knum,C1_121_tau,G11_iom,11,SigDMFT1,mu)#C1_121=C2_121!
            Sig3_2_112=mpilib.precalcsigp_mpi(U,beta,knum,C2_112_tau,G12_iom,12,SigDMFT1,mu)
            Sig3_11=Sig3_2_111+Sig3_1_111+Sig3_1_121+Sig3_2_121
            Sig3_12=Sig3_1_112+Sig3_2_112
        # update: Sigma_new=Sigma(DMFT)+Sigma(pert,first nth order)-Sigma(imp, first nth order)
        if rank ==0:
            Sig2_22=-Sig2_11.conjugate()
            # print('\t(Hartree11-oldHartreeloc)*U/beta=',(Hartree11-oldHartreeloc)*U/beta)
            # print('eff_Delta=',delta_eff)
            new_Sigma11=ori_Sigma11[:,None,None,None]+Sig2_11-sigimp11[:,None,None,None]#+(Hartree11-oldHartreeloc)*U/beta
            new_Sigma22=ori_Sigma22[:,None,None,None]+Sig2_22-sigimp22[:,None,None,None]#-(Hartree11-oldHartreeloc)*U/beta
            new_Sigma12=Sig2_12
            if order==3:
                Sig3_22=-Sig3_11.conjugate()
                new_Sigma11+=Sig3_11
                new_Sigma22+=Sig3_22
                new_Sigma12+=Sig3_12
            timei=time.time()
            diff=diff_sigma(Sigma11,new_Sigma11,Sigma12,new_Sigma12)
            # print('it=',it, 'diff=',diff,'time=',timei-time0,'memory usage:',output_memory_usage(),'MB')
            print(f'\tit={it},\tdiff={diff:.7f},\ttime={timei-time0:.2f}s,\tRAM usage={output_memory_usage():.2f} MB')#\tdelta_n11={(Hartree11/beta-Hartreeloc/beta):.9f},

            # plt.plot(SigDMFT1.real,label='SigDMFT1 real')
            # plt.plot(SigDMFT1.imag,label='SigDMFT1 imag') 
            # plt.plot(Sigma12[nfreq:nfreq+freqdisplayed,0,0,0].real,label='Sigma12 {} real'.format(it))
            plt.plot(Sigma11[nfreq:nfreq+freqdisplayed,0,0,0].real,label='Sigma11 {} real'.format(it))
            plt.plot(Sigma11[nfreq:nfreq+freqdisplayed,0,0,0].imag,label='Sigma11 {} imag'.format(it))
            # plt.plot(Sigma22[nfreq:nfreq+freqdisplayed,0,0,0].real,label='Sigma22 {} real'.format(it))
            # plt.plot(Sigma22[nfreq:nfreq+freqdisplayed,0,0,0].imag,label='Sigma22 {} imag'.format(it))
            # plt.plot(Sig1_11[nfreq:nfreq+freqdisplayed,0,0,0].real+Sig2_11[nfreq:nfreq+freqdisplayed,0,0,0].real,label='sig12_11 {} real'.format(it))
            # plt.plot(Sig1_11[nfreq:nfreq+freqdisplayed,0,0,0].imag+Sig2_11[nfreq:nfreq+freqdisplayed,0,0,0].imag,label='sig12_11 {} imag'.format(it))
            # plt.plot(sig2_11[nfreq:,0,0,0].real,label='sig2_11 {} real'.format(it))
            # plt.plot(sig2_22[nfreq:,0,0,0].imag,label='sig2_22 {} imag'.format(it))
            # plt.plot(Sig3_11[nfreq:,0,0,0].real,label='Sig3_11 {} real'.format(it))
            # plt.plot(Sig3_11[nfreq:,0,0,0].imag,label='Sig3_11 {} imag'.format(it))            
            # plt.plot(sigimp11[nfreq:].real,label='sigimp11 {} real'.format(it))
            # plt.plot(sigimp11[nfreq:].imag,label='sigimp11 {} imag'.format(it))
            # plt.legend()
            # plt.grid()
            # plt.show()
            Sigma11=new_Sigma11
            Sigma22=new_Sigma22
            Sigma12=new_Sigma12
        
        diff=comm.bcast(diff, root=0)
        delta_eff=comm.bcast(delta_eff, root=0)
        delta_inf=comm.bcast(delta_inf, root=0)
        # print('it=',it,'diff=',diff)
        comm.Bcast(Sigma11, root=0)
        comm.Bcast(Sigma22, root=0)
        comm.Bcast(Sigma12, root=0)
        it+=1
    comm.Barrier()
    # print('iteration finished!')
    if rank ==0:
        print('\textramu=',(delta_eff-delta_inf)/delta_inf,'delta')
        # plt.plot(SigDMFT1.real,label='SigDMFT1 real')
        # plt.plot(SigDMFT1.imag,label='SigDMFT1 imag') 
        plt.legend()
        plt.grid()
        plt.show()
    return Sigma11,Sigma22,Sigma12,(delta_eff-delta_inf)

def Delta_pert_DMFT(SigA,SigB,U,T,knum,nfreq,order):
    mu=U/2
    beta=1/T
    om = (2*np.arange(nfreq)+1)*np.pi/beta
    iom= 1j*om
    sig_new_11,sig_new_22,sig_new_12,extramu=iterative_perturbation(SigA,SigB,U,T,nfreq,order)
    if rank==0:
        Delta0_11,Delta0_22=Delta_DMFT(SigA,SigB,U,T,knum)
        
        Delta_11,Delta_22=Delta_DMFT(SigA+extramu,SigB-extramu,U,T,knum)

        # sig_imp_new_11=np.sum(sig_new_11,axis=(1,2,3))/knum**3
        # sig_imp_new_22=np.sum(sig_new_22,axis=(1,2,3))/knum**3
        # z_1=z4D(beta,mu,sig_new_11,knum,nfreq)#z-delta
        # z_2=z4D(beta,mu,sig_new_22,knum,nfreq)#z+delta
        # #final GF also have this extramu...? No! this is perturbed GF.
        # Gk_new_11,Gk_new_12=G_iterative(knum,z_1,z_2,sig_new_12)
        # Gk_new_22=-Gk_new_11.conjugate()
        # Gk_imp_new_11=np.sum(Gk_new_11,axis=(1,2,3))/knum**3
        # Gk_imp_new_22=np.sum(Gk_new_22,axis=(1,2,3))/knum**3
        # Delta_11=iom+mu-sig_imp_new_11[nfreq:]-1/Gk_imp_new_11[nfreq:]
        # Delta_22=iom+mu-sig_imp_new_22[nfreq:]-1/Gk_imp_new_22[nfreq:]
        if sig_plot==1:
            plt.plot(Delta_11[:].real,label='pert3_11 real')
            plt.plot(Delta_11[:].imag,label='pert3_11 imag')
            plt.plot(Delta_22[:].real,label='pert3_22 real')
            plt.plot(Delta_22[:].imag,label='pert3_22 imag')
            plt.plot(Delta0_11[:].real,label='DMFT11 real')
            plt.plot(Delta0_11[:].imag,label='DMFT11 imag')
            plt.plot(Delta0_22[:].real,label='DMFT22 real')
            plt.plot(Delta0_22[:].imag,label='DMFT22 imag')
            plt.title('Hybridization: U={},T={}'.format(U,T))
            plt.legend()
            plt.grid()
            plt.show()
        # print('plot closed!')
        if sig_plot==0:# run mode, output
            f = open(fileD, 'w')
            for i,iom in enumerate(om):
                print(iom, Delta_11[i].real, Delta_11[i].imag, Delta_22[i].real, Delta_22[i].imag, file=f) 
            f.close()
            print('delta file saved!')
    return 0





if __name__ == "__main__":
    fileS = 'Sig.OCA'
    fileD= 'Delta.inp'
    fileS12='Sig12.dat'
    knum=10 # default
    sig_plot=0  #1=plot 0= do not plot
    pltkpts=1    # max:47 for knum=10
    freqdisplayed=50 # only show first # Matsubara freqs
    ordernum=2# order of perturbation
    if (len(sys.argv)==1):# this is for test
        # standard test
        if rank==0:
            print('This is test mode')
        sig_plot=1# in the test mode, plot the sigma. in the import/calling mode, do not plot.
        # U=10.0  
        # T=0.473        
        # U=7.0  
        # T=0.4
        U=5.0  
        T=0.26       
        # U=3.0  
        # T=0.12   
        knum=10
        nfreq=500
        
        index=49#index start from 1, not 0
        sigma=np.loadtxt('../files_boldc/{}_{}_{}/Sig.out.{}'.format(0,U,T,index))[:nfreq,:]
        # sigma=np.loadtxt('../files_boldc/0_{}_{}/Sig.OCA.{}'.format(U,T,index))[:nfreq,:]
        # sigma=np.loadtxt('../files_ctqmc/{}_{}/ori_Sig.out.{}'.format(U,T,index))[:nfreq,:]
        # sigma=np.loadtxt('../files_pert_ctqmc/{}_{}/Sig.out.{}'.format(U,T,index))[:nfreq,:]
        # sigA=sigma[:,1]+1j*sigma[:,2]#sig+delta
        # sigB=U-sigma[:,1]+1j*sigma[:,2]#sig+delta
        # sigB=sigma[:,3]+1j*sigma[:,4]#sig-delta
        sigA=(+U/2+0.001)*np.ones(nfreq,dtype=complex)+1j*sigma[:,2]#sig+delta
        sigB=(+U/2-0.001)*np.ones(nfreq,dtype=complex)+1j*sigma[:,2]#sig-delta
        if sigma[-1,1]<sigma[-1,3]:
            sigA=sigma[:,3]+1j*sigma[:,2]#sig+delta
            sigB=sigma[:,1]+1j*sigma[:,2]#sig-delta
        # iterative_perturbation(sigA,sigB,U,T,nfreq,ordernum)
        Delta_pert_DMFT(sigA,sigB,U,T,knum,nfreq,ordernum)
        # plt.close('all')
    # collect command line parameters
    # In actual calcs we use more parameters to control.
    else:# this is for actual getting data
        nfreq=500
        order=2
        if (len(sys.argv)>=4):
            myorder=int(sys.argv[1])
            U=float(sys.argv[2])
            T=float(sys.argv[3])
        if (len(sys.argv)>=5):
            fileS=sys.argv[4]
        if (len(sys.argv)>=6):
            fileD=sys.argv[5]
        if (len(sys.argv)>=7):
            knum=int(sys.argv[6])
        if (len(sys.argv)>=8) or (len(sys.argv)<3):
            if rank ==0:
                print('input format does not match!\n format: mpirun -np 8 python perturb_mpi.py U T sigfile deltafile knum\nsigfile deltafile knum are optional')
                print('example: mpirun -np 8 python perturb.py 7.0 0.38 Sig.dat')

        if rank==0:
            print('-----------Perturbed Iteration of DMFT------')
            print('order=',myorder,' T=',T,' U=',U,' knum=',knum,' sigfile=',fileS,' deltafile=',fileD)

        if (os.path.exists(fileS)):
            Sf = np.loadtxt(fileS).T
            sigA = Sf[1,:]+Sf[2,:]*1j
            sigB = U-Sf[1,:]+Sf[2,:]*1j
            if Sf[1,-1]<Sf[3,-1]:
                sigA=Sf[3,:]+1j*Sf[4,:]#sig+delta
                sigB=U-Sf[3,:]+1j*Sf[4,:]#sig-delta
        else:
            if rank==0:
                print('cannot find {}!'.format(fileS))
        Delta_pert_DMFT(sigA,sigB,U,T,knum,nfreq,myorder)
