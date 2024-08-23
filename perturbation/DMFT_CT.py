import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess,math
from scipy.interpolate import interp1d
import time
sys.path.append('../python_src/')
from mpi4py import MPI
import perturb_lib as lib
import perturb_imp as imp
import fft_convolution as fft
import diagrams
import mpi_module
import copy
import perm_def
sys.path.append('../python_src/diagramsMC/')
import basis
import svd_diagramsMC_cutPhi
import diag_def_cutPhifast
import imp_diag_def
import imp_svd_diagramsMC

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

'''
This is a python script which generate all DMFT diagrams of different orders (up to order 4) and save them using the svd basis.
'''
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
    # filename1='../files_variational/{}_{}_{}/Sig.out.{}'.format(B,U,T,index)
    # filename2='../files_variational/{}_{}_{}/Sig.OCA.{}'.format(B,U,T,index)
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

def get_gtauloc(U,T,knum=10,nfreq=500):
    beta=1/T
    mu=U/2
    name1='../files_boldc/{}_{}/Sig.out'.format(U,T)
    filename1=readDMFT(name1)
    name2='../files_boldc/{}_{}/Sig.OCA'.format(U,T)
    filename2=readDMFT(name2)
    name3='../files_ctqmc/{}_{}/Sig.out'.format(U,T)
    filename3=readDMFT(name3)
    # print(filename1)
    # print(filename2)
    if (os.path.exists(filename1)):
        filename=filename1
    elif (os.path.exists(filename2)):
        filename=filename2
        # print('reading DMFT data from {}'.format(filename))
    elif (os.path.exists(filename3)):
        filename=filename3
    else:
        print('these 3 filenames cannot be found:\n {} \n {} \n {}\n'.format(name1,name2,name3))  
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
    Sigma11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    Sigma11+=lib.ext_sig(sigA)[:,None,None,None]
    Sigma22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    Sigma22+=lib.ext_sig(sigB)[:,None,None,None]
    Sigma12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    z_1=lib.z4D(beta,mu,Sigma11,knum,nfreq)#z-delta
    z_2=lib.z4D(beta,mu,Sigma22,knum,nfreq)#z+delta
    G11_iom,G12_iom=lib.G_iterative(knum,z_1,z_2,Sigma12)
    G22_iom=-G11_iom.conjugate()
    G11imp_iom=np.sum(G11_iom,axis=(1,2,3))/knum**3 # impurity GF=sum_k DMFT GF
    G22imp_iom=np.sum(G22_iom,axis=(1,2,3))/knum**3 # impurity GF=sum_k DMFT GF
    G11_tau=fft.fermion_fft_diagG(knum,G11_iom,beta,sigA,mu)# currently sigma12=0
    G12_tau=fft.fast_ft_fermion(G12_iom,beta)
    G22_tau=G11_tau[::-1] 
    # Gloc11_tau=np.sum(G11_tau,axis=(1,2,3))[:,None,None,None]/knum**3*np.ones((knum,knum,knum))[None,:,:,:]
    # Gloc22_tau=np.sum(G22_tau,axis=(1,2,3))[:,None,None,None]/knum**3*np.ones((knum,knum,knum))[None,:,:,:]
    Gloc11=np.sum(G11_tau,axis=(1,2,3))/knum**3
    # Gloc22=np.sum(G22_tau,axis=(1,2,3))/knum**3
    Gloc22=Gloc11[::-1]
    return G11imp_iom,G22imp_iom,Gloc11.real,Gloc22.real,G11_tau.real,G12_tau.real,G22_tau.real

def gen_allCTs(U,T,lmax=10):
    '''
    generate all cts. it is saved in a file like this:
    here i only give sig11. since sig12 can be figured out by particle-hole symmetry.
    1st column:                                   after 1st column:         
    (lmax)                                       (svd coefficient of 2nd, 3rd, 4th order CTs)
    (number of svd time points)
    (number of svd kernel freq)
    (the const, the 1st order, hartree)
    where time and freq list in the kernal are defined as:
    taulist=(np.arange(taunum+1))/taunum*beta#
    omlist=(2*np.arange(2*nfreq)+1-2*nfreq)*np.pi/beta 
    ker=basis.fermi_kernel(taulist,omlist,beta)
    
    Even all these quantities are given, the svd basis u is still not fixed. so we also have to save the svd basis.
    The basis is lmax*taunum, which should be able to be saved in a file.
    '''
    beta=1/T
    knum=10
    nfreq=500
    # about the basis
    taunum=50
    taulist=(np.arange(taunum+1))/taunum*beta#
    omlist=(2*np.arange(2*nfreq)+1-2*nfreq)*np.pi/beta 
    ker=basis.fermi_kernel(taulist,omlist,beta)
    ut=np.empty((lmax,taunum+1),dtype=float)
    filename_u='./Sigma_imp/taubasis.txt'
    if rank==0:
        print('U={},T={}'.format(U,T))
        if (os.path.exists(filename_u))==0:# generate ut and save it
            ut,sig,v=basis.svd_kernel_fast(ker,lmax)
            np.savetxt(filename_u,ut.T)
            print('generate new ut')
        else:
            print('use saved ut')
        # if diag this is all coeffs. if offdiag, we only need 1st, 3rd, 5th,...
        # print('sigular values:',sig)
    # if (os.path.exists(filename_u)):# already has this ut basis
    ut=np.loadtxt(filename_u).T
    
    # ut = np.ascontiguousarray(ut)
    # comm.Bcast(ut, root=0)

    # kbasis
    imax=8
    kbasis=np.empty((imax,knum,knum,knum),dtype=float)
    if rank==0:
        kbasis=basis.gen_kbasis(imax,knum)
    kbasis = np.ascontiguousarray(kbasis)
    comm.Bcast(kbasis, root=0)    

    sublatind_basis=np.array([[1,0,0,0],
                              [0,1,0,0],
                              [0,0,1,0],
                              [0,0,0,1]])
    # sublatind_basis=np.array([[1,1,1,1],
    #                           [1,-1,1,-1],
    #                           [1,1,-1,-1],
    #                           [1,-1,-1,1]])/2# Hadamard matrix. this might be better.


    all_returns=get_gtauloc(U,T)
    if all_returns==0:
        return 0
    G11imp_iom,G22imp_iom,G11imp_tau,G22imp_tau,G11_tau,G12_tau,G22_tau=all_returns
    # G124Dempty=np.zeros((2*nfreq,knum,knum,knum))
    # G114D=np.ones((2*nfreq,knum,knum,knum))*G11imp_tau[:,None,None,None]
    # G224D=np.ones((2*nfreq,knum,knum,knum))*G22imp_tau[:,None,None,None]
    GFs=(G11imp_tau,G22imp_tau)
    GFdispersive=(G11_tau,G12_tau,G22_tau)
    if rank==0:
        # getting easy diagrams through BF.
        n0loc11=lib.particlenumber1D(G11imp_iom,beta)
        n0loc22=lib.particlenumber1D(G22imp_iom,beta)
        sigimp_1=n0loc11*U# this is a bug!!!!!!!!!   I don't want to rerun it so later it will be noted as sigimp1_22
        sigimp_2=imp.pertimp_func_tau(G11imp_tau,beta,U,knum,2)
        sigimp_31,sigimp_32=imp.pertimp_func_tau(G11imp_tau,beta,U,knum,3)
        sigimp_41,sigimp_42,sigimp_45=imp.pertimp_func_tau(G11imp_tau,beta,U,knum,4)
        ori_grid=(np.arange(nfreq*2)+0.5)/(nfreq*2)*beta
        interpolator_2 = interp1d(ori_grid, sigimp_2, kind='cubic', fill_value='extrapolate')
        interpolator_31 = interp1d(ori_grid, sigimp_31, kind='cubic', fill_value='extrapolate')
        interpolator_32 = interp1d(ori_grid, sigimp_32, kind='cubic', fill_value='extrapolate')
        interpolator_41 = interp1d(ori_grid, sigimp_41, kind='cubic', fill_value='extrapolate')
        interpolator_42 = interp1d(ori_grid, sigimp_42, kind='cubic', fill_value='extrapolate')
        interpolator_45 = interp1d(ori_grid, sigimp_45, kind='cubic', fill_value='extrapolate')
        cl2=basis.inner_prod(interpolator_2(taulist),ut)
        cl31=basis.inner_prod(interpolator_31(taulist),ut)
        cl32=basis.inner_prod(interpolator_32(taulist),ut)
        cl41=basis.inner_prod(interpolator_41(taulist),ut)
        cl42=basis.inner_prod(interpolator_42(taulist),ut)
        cl45=basis.inner_prod(interpolator_45(taulist),ut)

    sublatindinit4=np.array([1,2,1,2,1,2,1,2])
    # definition of integrands. some are not actually used but just to check if the BF matches the result of MC.
    # func31=imp_diag_def.FuncNDiagNew(T,U,knum,taunum,nfreq,3,ut,kbasis,sublatind_basis,perm_def.perm31,GFs,perm_def.dep31,6)
    # func32=imp_diag_def.FuncNDiagNew(T,U,knum,taunum,nfreq,3,ut,kbasis,sublatind_basis,perm_def.perm32,GFs,perm_def.dep32,6)
    func41=imp_diag_def.FuncNDiagNew(T,U,taunum,nfreq,4,ut,perm_def.perm41,GFs,perm_def.dep41,sublatindinit4,8)
    func42=imp_diag_def.FuncNDiagNew(T,U,taunum,nfreq,4,ut,perm_def.perm42,GFs,perm_def.dep42,sublatindinit4,8)
    # the 3rd diagram has the symmetry factor 2, since cutting the diagram will generate 4 kinds of sigma diagrams.
    func43=imp_diag_def.FuncNDiagNew(T,U,taunum,nfreq,4,ut,perm_def.perm43,GFs,perm_def.dep43,sublatindinit4,2)
    func44=imp_diag_def.FuncNDiagNew(T,U,taunum,nfreq,4,ut,perm_def.perm44,GFs,perm_def.dep44,sublatindinit4,4)
    func45=imp_diag_def.FuncNDiagNew(T,U,taunum,nfreq,4,ut,perm_def.perm45,GFs,perm_def.dep45,sublatindinit4,8)
    
    p = params()
    # cl31_MC=imp_svd_diagramsMC.Summon_Integrate_Parallel_impurity(func31,p,p,lmax,ut)
    # cl32_MC=imp_svd_diagramsMC.Summon_Integrate_Parallel_impurity(func32,p,p,lmax,ut)
    # cl41_MC=imp_svd_diagramsMC.Summon_Integrate_Parallel_impurity(func41,p,lmax,ut)
    # cl42_MC=imp_svd_diagramsMC.Summon_Integrate_Parallel_impurity(func42,p,lmax,ut)
    cl43=imp_svd_diagramsMC.Summon_Integrate_Parallel_impurity(func43,p,lmax,ut)
    cl44=imp_svd_diagramsMC.Summon_Integrate_Parallel_impurity(func44,p,lmax,ut)
    # cl45_MC=imp_svd_diagramsMC.Summon_Integrate_Parallel_impurity(func45,p,lmax,ut)
    

    # for testing dispersive version
    # func44dis=diag_def_cutPhifast.FuncNDiagNew(T,U,knum,taunum,nfreq,4,ut,kbasis,sublatind_basis,perm_def.perm44,GFdispersive,perm_def.dep44,4)
    # a,b,c,sig44tau_test=svd_diagramsMC_cutPhi.Summon_Integrate_Parallel_dispersive(func44dis,p,imax,lmax,ut,kbasis,sublatind_basis,beta)
    # func43dis=diag_def_cutPhifast.FuncNDiagNew(T,U,knum,taunum,nfreq,4,ut,kbasis,sublatind_basis,perm_def.perm43,GFdispersive,perm_def.dep43,2)
    # a,b,c,sig43tau_test=svd_diagramsMC_cutPhi.Summon_Integrate_Parallel_dispersive(func43dis,p,imax,lmax,ut,kbasis,sublatind_basis,beta)
    if rank==0:
    #     # these are testing
    #     plt.plot(cl44,label='BF')
    #     # plt.plot(cl42_MC,label='MCimp')
    #     plt.legend()
    #     plt.show()
    #     BFres44=basis.restore_Gf(cl44,ut)
    #     BFres43=basis.restore_Gf(cl43,ut)
    #     # MCres1=basis.restore_Gf(cl43_MC,ut)
    #     for kx in np.arange(knum):
    #         for ky in np.arange(knum):
    #             for kz in np.arange(knum):
    #                 plt.plot(BFres44,label='BF 44')
    #                 # plt.plot(MCres1,label='MCimp')
    #                 plt.plot(sig44tau_test[:,kx,ky,kz],label='MCdisp ave 44')
    #                 plt.legend()
    #                 plt.show()     
    #                 plt.plot(BFres43,label='BF 43')
    #                 # plt.plot(MCres1,label='MCimp')
    #                 plt.plot(sig43tau_test[:,kx,ky,kz],label='MCdisp ave 43')
    #                 plt.legend()
    #                 plt.show()  
    #     # plt.plot(BFres/MCres1,label='ratio')
    #     # plt.legend()
    #     # plt.show()     


        # 1 col for 1st, 1col for 2nd, 2cols for 3rd, 5cols for 4th. 1+1+2+5=9
        outarray=np.zeros((max(4,lmax),9))
        outarray[0,0]=lmax
        outarray[1,0]=taunum
        outarray[2,0]=nfreq
        outarray[3,0]=sigimp_1
        outarray[:lmax,1]=cl2
        outarray[:lmax,2]=cl31
        outarray[:lmax,3]=cl32
        outarray[:lmax,4]=cl41
        outarray[:lmax,5]=cl42
        outarray[:lmax,6]=cl43
        outarray[:lmax,7]=cl44
        outarray[:lmax,8]=cl45
        coeff_file='./Sigma_imp/coeff_{}_{}.txt'.format(U,T)
        np.savetxt(coeff_file,outarray)



    return 0


def run_imp():
    T_bound=np.array(((3.0,0.08,0.14),(5.,0.2,0.31),
                      (8.,0.4,0.6),(10.,0.5,0.63),(12.,0.4,0.6),(14.,0.4,0.5)))
    #(4.,0.1,0.25),(6.,0.27,0.37),(7.,0.27,0.4),(9.,0.28,0.45),(11.,0.3,0.5),
    for list in T_bound:
        U=list[0]
        # print(U)
        
        for T in np.arange(int(list[1]*100),int(list[2]*100))/100:
            filename='./Sigma_imp/coeff_{}_{}.txt'.format(U,T)
            if (os.path.exists(filename))==0:
                if rank==0:
                    print(U,T)
                gen_allCTs(U,T)
            else:
                if rank==0:
                    print('skip U={} T={}'.format(U,T))



if __name__ == "__main__":
    U=10.
    T=0.05
    # Tlist8=np.array([0.05,0.25])
    # Tlist10=np.array([0.1,0.25,0.28,0.31,0.35,0.4,0.45,0.5])
    # for T in Tlist10:
    #     gen_allCTs(U,T)
    # gen_allCTs(8.0,0.3)
    # gen_allCTs(10.0,0.3)
    # gen_allCTs(10.0,0.36)
    # gen_allCTs(10.0,0.25)
    run_imp()