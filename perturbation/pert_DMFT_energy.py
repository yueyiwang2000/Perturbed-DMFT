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
import DMFT_CT
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

'''
Yueyi Wang, Aug 2024
This code aims to read self-energy from Sigma_disp folder to get total energy done.
For the total energy, which is easier, use Midgal-Galiski formula.
Later on we will have an algorithm for Free energy, which is much more complicated.
'''
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
    return ut,lmax,taunum,nfreq,sigimp_1,Sigmaimpiom2_11,Sigmaimpiom31_11,Sigmaimpiom32_11,Sigmaimpiom41_11,Sigmaimpiom42_11,Sigmaimpiom43_11,Sigmaimpiom44_11,Sigmaimpiom45_11


def read_complex_numbers(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    complex_numbers = []
    for line in lines:
        line = line.strip()
        
        complex_number = complex(line)
        complex_numbers.append(complex_number)

    complex_array = np.array(complex_numbers)
    
    return complex_array


def get_sigma_and_G(U,T,order,alpha):
    knum=10
    beta=1/T
    mu=U/2
    sigmafilename11='./Sigma_disp/{}_{}/{}_{}_{}_{}_11.dat'.format(U,T,U,T,order,alpha)
    sigmafilename11const='./Sigma_disp/{}_{}/{}_{}_{}_{}_11const.dat'.format(U,T,U,T,order,alpha)
    sigmafilename12='./Sigma_disp/{}_{}/{}_{}_{}_{}_12.dat'.format(U,T,U,T,order,alpha)
    filenameu='./Sigma_imp/taubasis.txt'
       
    ut=np.loadtxt(filenameu).T 
    taunum=np.shape(ut)[1]-1
    nfreq=500
    if order>0:
        cli_11=np.loadtxt(sigmafilename11)
        cli_12=np.loadtxt(sigmafilename12)
        ci_11=np.loadtxt(sigmafilename11const)# for const
        
        imax=np.shape(cli_11)[1]
        kbasis=np.empty((imax,knum,knum,knum),dtype=float)
        if rank==0:
            kbasis=basis.gen_kbasis(imax,knum)
        kbasis = np.ascontiguousarray(kbasis)
        comm.Bcast(kbasis, root=0)    
        # print('imax=',imax)
        

        taulist=(np.arange(taunum+1))/taunum*beta
        ori_grid=(np.arange(nfreq*2)+0.5)/(nfreq*2)*beta
        Sig11tk_raw=basis.restore_tk(cli_11,ut,kbasis)
        Sig12tk_raw=basis.restore_tk(cli_12,ut,kbasis)
        interpolator_11 = interp1d(taulist, Sig11tk_raw, kind='cubic', axis=0, fill_value='extrapolate')
        interpolator_12 = interp1d(taulist, Sig12tk_raw, kind='cubic',  axis=0,fill_value='extrapolate')
        Sig11tk_full=interpolator_11(ori_grid)
        Sig12tk_full=interpolator_12(ori_grid)
        sig11const=basis.restore_k(ci_11,kbasis)
        Sigiom_11=fft.fast_ift_fermion(Sig11tk_full,beta)+sig11const[None,:,:,:]
        Sigiom_12=fft.fast_ift_fermion(Sig12tk_full,beta)
        # for kx in np.arange(knum):
        #     for ky in np.arange(knum):
        #         for kz in np.arange(knum):
        #             # plt.plot(Sigiom_11[:,kx,ky,kz].real,label='11 real')
        #             # plt.plot(Sigiom_11[:,kx,ky,kz].imag,label='11 imag')
        #             plt.plot(Sig11tk_raw[:,kx,ky,kz],label='tau')
        #             plt.legend()
        #             plt.show()


    elif order==0:# if order==0, the self-energy is not saved using the basis, since it is local.
        Sigiom_11=read_complex_numbers(sigmafilename11)[:,None,None,None]*np.ones((2*nfreq,knum,knum,knum))
        # Sigiom_11=np.loadtxt(sigmafilename11)[:,None,None,None]*np.zeros((2*nfreq,knum,knum,knum))
        Sigiom_12=np.zeros((2*nfreq,knum,knum,knum))
    Sigiom_22=U-Sigiom_11.conjugate()
    znew_1=z4D(beta,mu,Sigiom_11,knum,nfreq)
    znew_2=z4D(beta,mu,Sigiom_22,knum,nfreq)
    Gdress11_iom,Gdress12_iom=G_iterative(knum,znew_1,znew_2,Sigiom_12)
    Gdress22_iom=-Gdress11_iom.conjugate()
    return Sigiom_11,Sigiom_12,Sigiom_22,Gdress11_iom,Gdress12_iom,Gdress22_iom


def mag_test(U,T):
    '''
    This function is for testing. To check the self-energy stored in the form of coefficients can really be restored and give the same magnetization.
    '''
    beta=1/T
    maxorder=4
    alpha_arr=np.array(([0.05,0.1,0.2,0.3,0.6,1.0]))
    alphanum=np.shape(alpha_arr)[0]
    mag_arr=np.zeros((alphanum,maxorder+1))
    for order in np.arange(maxorder+1):
        for ialp,alpha in enumerate(alpha_arr):
            Sigiom_11,Sigiom_12,Sigiom_22,Gdress11_iom,Gdress12_iom,Gdress22_iom=get_sigma_and_G(U,T,order,alpha)
            # znew_1=z4D(beta,mu,sigfinal11,knum,nfreq)
            # znew_2=z4D(beta,mu,sigfinal22,knum,nfreq)
            # Gdress11_iom,Gdress12_iom=G_iterative(knum,znew_1,znew_2,sigfinal12)
            # Gdress22_iom=-Gdress11_iom.conjugate()
            nnewloc11=particlenumber4D(Gdress11_iom,beta)
            nnewloc22=particlenumber4D(Gdress22_iom,beta)
            mag_arr[ialp,order]=nnewloc22-nnewloc11
    plt.plot(alpha_arr, mag_arr[:,0], marker='o', linestyle='-',label='0th')
    plt.plot(alpha_arr, mag_arr[:,1], marker='^', linestyle='-',label='1st')
    plt.plot(alpha_arr, mag_arr[:,2], marker='s', linestyle='-',label='2nd')
    plt.plot(alpha_arr, mag_arr[:,3], marker='p', linestyle='-',label='3rd')
    plt.plot(alpha_arr, mag_arr[:,4], marker='h', linestyle='-',label='4th')
    plt.legend()
    plt.title('mag vs Order: U={} T={}'.format(U, T))
    plt.show()
    return 0


def total_energy(U,T,order,alpha):
    '''
    previously we had a python script to calculate the total energy and self energy, and save it in a file. 
    But now total energy can be calculated through the saved self-energy, which does not take too much time.
    However, Free energy is sum of diagrams of interacted GFs, which we haven't evaluate before. So free energy takes a lot more time.
    '''
    mu=U/2
    beta=1/T
    knum=10
    nfreq=500
    Sigiom_11,Sigiom_12,Sigiom_22,Gdress11_iom,Gdress12_iom,Gdress22_iom=get_sigma_and_G(U,T,order,alpha)
    s11_oo = Sigiom_11[-1,:,:,:].real# currently this is a 3d array, each k point has a s_oo.
    EimpS11 = -mu+s11_oo # this is also a 3d array. G~1/(iom-eimp), so we need eimp.
    s22_oo = Sigiom_22[-1,:,:,:].real
    EimpS22 = -mu+s22_oo
    # G11_tau=fft.fermion_fft_diagG_4D(knum,Gdress11_iom,beta,EimpS11)
    # G12_tau=fft.fast_ft_fermion(Gdress12_iom,beta)
    # G22_tau=fft.fermion_fft_diagG_4D(knum,Gdress22_iom,beta,EimpS22)
    H0_G=energy.H0G(Gdress12_iom,T,U)
    om= (2*np.arange(nfreq)+1)*np.pi/beta
    n11=(np.sum(Gdress11_iom).real/knum**3/beta+1/2)
    n22=(np.sum(Gdress22_iom).real/knum**3/beta+1/2)
    TrSigmaG=energy.fTrSigmaG(om, Gdress11_iom[nfreq:], Sigiom_11[nfreq:], EimpS11, beta,knum)+energy.fTrSigmaG(om, Gdress22_iom[nfreq:], Sigiom_22[nfreq:], EimpS22, beta,knum)+energy.fTrSigmaG_bf(om, Gdress12_iom[nfreq:], Sigiom_12[nfreq:], np.zeros((nfreq,knum,knum,knum)), beta,knum)*2
    TrSigmaG+=np.sum(n11*s11_oo+n22*s22_oo)/knum**3 # remember to add the infinite part!
    Edisp=H0_G+TrSigmaG/2
    # check impurity total energy.

    return Edisp

def total_energy_DMFT(U,T):
    mu=U/2
    beta=1/T
    knum=10
    nfreq=500

    beta=1/T
    mu=U/2
    name1='../files_boldc/{}_{}/Sig.out'.format(U,T)
    filename1=DMFT_CT.readDMFT(name1)
    name2='../files_boldc/{}_{}/Sig.OCA'.format(U,T)
    filename2=DMFT_CT.readDMFT(name2)
    name3='../files_ctqmc/{}_{}/Sig.out'.format(U,T)
    filename3=DMFT_CT.readDMFT(name3)
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
    Sigma11+=ext_sig(sigA)[:,None,None,None]
    Sigma22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    Sigma22+=ext_sig(sigB)[:,None,None,None]
    Sigma12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    z_1=z4D(beta,mu,Sigma11,knum,nfreq)#z-delta
    z_2=z4D(beta,mu,Sigma22,knum,nfreq)#z+delta
    G11_iom,G12_iom=G_iterative(knum,z_1,z_2,Sigma12)
    G22_iom=-G11_iom.conjugate()
    G11imp_iom=np.sum(G11_iom,axis=(1,2,3))/knum**3 # impurity GF=sum_k DMFT GF
    G22imp_iom=np.sum(G22_iom,axis=(1,2,3))/knum**3 # impurity GF=sum_k DMFT GF

    s11_oo = Sigma11[-1,:,:,:].real# currently this is a 3d array, each k point has a s_oo.
    EimpS11 = -mu+s11_oo # this is also a 3d array. G~1/(iom-eimp), so we need eimp.
    s22_oo = Sigma22[-1,:,:,:].real
    EimpS22 = -mu+s22_oo

    H0_G=energy.H0G(G12_iom,T,U)
    om= (2*np.arange(nfreq)+1)*np.pi/beta
    n11=(np.sum(G11_iom).real/knum**3/beta+1/2)
    n22=(np.sum(G22_iom).real/knum**3/beta+1/2)
    TrSigmaG=energy.fTrSigmaG(om, G11_iom[nfreq:], Sigma11[nfreq:], EimpS11, beta,knum)+energy.fTrSigmaG(om, G22_iom[nfreq:], Sigma22[nfreq:], EimpS22, beta,knum)
    TrSigmaG+=np.sum(n11*s11_oo+n22*s22_oo)/knum**3 # remember to add the infinite part!
    EDMFT=H0_G+TrSigmaG/2
    return EDMFT

def energy_test(U,T,ifplot=0):
    beta=1/T
    maxorder=4
    alpha_arr=np.array(([0.05,0.1,0.2,0.3,0.6,1.0]))
    alphanum=np.shape(alpha_arr)[0]
    E_arr=np.zeros((alphanum,maxorder+1))
    for order in np.arange(maxorder+1):
        for ialp,alpha in enumerate(alpha_arr):
            E_arr[ialp,order]=total_energy(U,T,order,alpha)
    if ifplot==1:
        plt.plot(alpha_arr, E_arr[:,0], marker='o', linestyle='-',label='0th')
        plt.plot(alpha_arr, E_arr[:,1], marker='^', linestyle='-',label='1st')
        plt.plot(alpha_arr, E_arr[:,2], marker='s', linestyle='-',label='2nd')
        plt.plot(alpha_arr, E_arr[:,3], marker='p', linestyle='-',label='3rd')
        plt.plot(alpha_arr, E_arr[:,4], marker='h', linestyle='-',label='4th')
        plt.legend()
        plt.title('Total energy vs Alpha: U={} T={}'.format(U, T))
        plt.xlabel('alpha')
        plt.ylabel('Total energy')
        plt.show()

    return E_arr

def free_energy_test(U,T,ifplot=0):
    beta=1/T
    maxorder=3
    # alpha_arr=np.array(([0.01,0.05,0.1,0.15,0.2,0.3,0.4,0.6,0.8,1.0]))
    energy_dir='./energydata/{}_{}.dat'.format(U,T)
    data=np.loadtxt(energy_dir)
    alpha_arr=data[:,0]
    alphanum=np.shape(alpha_arr)[0]
    F_arr=np.zeros((alphanum,maxorder+1))
    for order in np.arange(maxorder+1):
        for ialp,alpha in enumerate(alpha_arr):
            F_arr[ialp,order]=data[ialp,order+1]

    if ifplot==1:
        plt.plot(alpha_arr, F_arr[:,0], marker='o', linestyle='-',label='0th')
        plt.plot(alpha_arr, F_arr[:,1], marker='^', linestyle='-',label='1st')
        plt.plot(alpha_arr, F_arr[:,2], marker='s', linestyle='-',label='2nd')
        plt.plot(alpha_arr, F_arr[:,3], marker='p', linestyle='-',label='3rd')
        # plt.plot(alpha_arr, F_arr[:,4], marker='h', linestyle='-',label='4th')
        plt.legend()
        plt.title('Free energy vs Alpha: U={} T={}'.format(U, T))
        plt.xlabel('alpha')
        plt.ylabel('Free energy')
        plt.show()
    return F_arr


def free_energy_with_errbar(U,order,TN=0):
    '''
    
    '''
    T_bound=np.array(((3.0,0.08,0.14),(5.,0.2,0.31),
        (8.,0.25,0.58),(10.,0.31,0.56),(12.,0.24,0.53),(14.,0.26,0.4)))
    Ufound=0
    for list in T_bound:
        if list[0]==U:
            listT=list
            Ufound=1
            break
    if Ufound==0:
        print('U not found!')
        return 0
    T_arr=np.arange(int(listT[1]*100),int(listT[2]*100))/100
    Fmax=np.zeros_like(T_arr)
    Fmin=np.zeros_like(T_arr)
    # Fimp=np.zeros_like(T_arr)
    for iT,T in enumerate(T_arr):
        E_array=free_energy_test(U,T)[:,order]
        Fmax[iT]=max(E_array[:5])
        Fmin[iT]=min(E_array[:5])
        if T<TN:
            Fmax[iT]=max(E_array[5:])
            Fmin[iT]=min(E_array[5:])
        # Eimp[iT]=total_energy_DMFT(U,T)
    F_ave=(Fmax+Fmin)/2
    F_err=(Fmax-Fmin)/2
    plt.errorbar(T_arr, F_ave, yerr=F_err, fmt='-o', capsize=5,color='r',label='3rd order')
    # plt.plot(T_arr,Fimp,color='b', linestyle='--',label='DMFT')
    plt.ylabel('free Energy')
    plt.xlabel('Temperature')
    plt.title('Temperature dependence of free energy: U={}'.format(U))
    plt.legend()
    plt.show()
    plt.plot((T_arr[:-1]+T_arr[1:])/2,(-F_ave[:-1]+F_ave[1:])/0.01,label='dF_pert/dT')
    # plt.plot((T_arr[:-1]+T_arr[1:])/2,(-Fimp[:-1]+Fimp[1:])/0.01,label='dF_DMFT/dT')
    plt.ylabel('dF/dT')
    plt.xlabel('Temperature')
    plt.legend()
    plt.show()    
    return 0



def energy_with_errbar(U,order,TN=0):
    '''
    
    '''
    T_bound=np.array(((3.0,0.08,0.14),(5.,0.2,0.31),
        (8.,0.2,0.59),(10.,0.25,0.63),(12.,0.24,0.53),(14.,0.26,0.4)))
    Ufound=0
    for list in T_bound:
        if list[0]==U:
            listT=list
            Ufound=1
            break
    if Ufound==0:
        print('U not found!')
        return 0
    T_arr=np.arange(int(listT[1]*100),int(listT[2]*100))/100
    Emax=np.zeros_like(T_arr)
    Emin=np.zeros_like(T_arr)
    Eimp=np.zeros_like(T_arr)
    for iT,T in enumerate(T_arr):
        E_array=energy_test(U,T)[:,order]
        Emax[iT]=max(E_array)
        Emin[iT]=min(E_array)
        if T<TN:
            Emin[iT]=min(E_array[3:])
        Eimp[iT]=total_energy_DMFT(U,T)
    E_ave=(Emax+Emin)/2
    E_err=(Emax-Emin)/2
    plt.errorbar(T_arr, E_ave, yerr=E_err, fmt='-o', capsize=5,color='r',label='4th order')
    plt.plot(T_arr,Eimp,color='b', linestyle='--',label='DMFT')
    plt.ylabel('Total Energy')
    plt.xlabel('Temperature')
    plt.title('Temperature dependence of total energy: U={}'.format(U))
    plt.legend()
    plt.show()
    plt.plot((T_arr[:-1]+T_arr[1:])/2,(-E_ave[:-1]+E_ave[1:])/0.01,label='dE_pert/dT')
    plt.plot((T_arr[:-1]+T_arr[1:])/2,(-Eimp[:-1]+Eimp[1:])/0.01,label='dE_DMFT/dT')
    plt.ylabel('dE/dT')
    plt.xlabel('Temperature')
    plt.legend()
    plt.show()    
    return 0

if __name__ == "__main__":
    # mag_test(3.,0.08)
    # energy_test(8.,0.25,1)
    # free_energy_test(8.,0.31,1)
    # energy_with_errbar(8.,4,0.1)
    free_energy_with_errbar(8.,3,0.3)