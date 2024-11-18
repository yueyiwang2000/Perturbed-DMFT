import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import numpy as np
import os,sys,subprocess,math
import time
sys.path.append('../python_src/')
from mpi4py import MPI
from perturb_lib import *
import perturb_imp as imp
import fft_convolution as fft
import pert_energy_lib as energy
# import perm_def
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
# import diagrams
# import mpi_module
# import serial_module
import copy
# sys.path.append('../python_src/diagramsMC/')
import diagramsMC.basis as basis
# import diagramsMC.dispersive_sig.svd_diagramsMC_cutPhi as svd_diagramsMC_cutPhi
# import diagramsMC.dispersive_sig.diag_def_cutPhifast as diag_def_cutPhifast
import DMFT_CT
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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


def get_valid_rows_indices(arr):
    '''
    array defind as [alp,order], at some alpha will be all none.
    '''
    return [i for i, row in enumerate(arr) if all(element is not None for element in row)]

def check_alpha(U,T,alphaarrraw,maxorder=4,searchopt=1):
    
    if searchopt==1:
        foldernum='_search'
    else:
        foldernum=1
    backup=''
    alphavalid=np.ones_like(alphaarrraw)
    for ialp,alpha in enumerate(alphaarrraw):
        for order in np.arange(maxorder+1):
            sigmafilename11='./Sigma_disp{}/{}{}_{}/{}_{}_{}_{}_11.dat'.format(foldernum,backup,U,T,U,T,order,alpha)
            sigmafilename11const='./Sigma_disp{}/{}{}_{}/{}_{}_{}_{}_11const.dat'.format(foldernum,backup,U,T,U,T,order,alpha)
            sigmafilename12='./Sigma_disp{}/{}{}_{}/{}_{}_{}_{}_12.dat'.format(foldernum,backup,U,T,U,T,order,alpha)
            check=0
            if order>0:
                if (os.path.exists(sigmafilename11))==0 or (os.path.exists(sigmafilename11const))==0 or (os.path.exists(sigmafilename12))==0:
                    check=1
            else:
                if (os.path.exists(sigmafilename11))==0:
                    check=1
            if check==1:
                alphavalid[ialp]=0
    indices = np.where(alphavalid == 1)[0]
    return alphaarrraw[indices]

def get_Tlist(U):
    T_bound=np.array(((4.0,0.1,0.5),(5.,0.2,0.31),
        (8.,0.2,2.0),(10.,0.25,0.63),(12.,0.2,0.6),(14.,0.26,0.5)))
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
    Tvalid=np.zeros_like(T_arr)
    for iT,T in enumerate(T_arr):
        dir='./Sigma_disp1/{}_{}'.format(U,T)
        if os.path.exists(dir):
            Tvalid[iT]=1
    indices = np.where(Tvalid == 1)[0]
    return T_arr[indices]


def get_sigma_and_G(U,T,order,alpha,foldernum=1,ifchecksvd=0):
    knum=10
    beta=1/T
    mu=U/2
    backup=''
    # backup='backup3/'
    sigmafilename11='./Sigma_disp{}/{}{}_{}/{}_{}_{}_{}_11.dat'.format(foldernum,backup,U,T,U,T,order,alpha)
    sigmafilename11const='./Sigma_disp{}/{}{}_{}/{}_{}_{}_{}_11const.dat'.format(foldernum,backup,U,T,U,T,order,alpha)
    sigmafilename12='./Sigma_disp{}/{}{}_{}/{}_{}_{}_{}_12.dat'.format(foldernum,backup,U,T,U,T,order,alpha)
    filenameu='./Sigma_imp/taubasis.txt'
       
    ut=np.loadtxt(filenameu).T 
    taunum=np.shape(ut)[1]-1
    nfreq=500
    imax=4
    if order>0:
        if (os.path.exists(sigmafilename11))==0 or (os.path.exists(sigmafilename11const))==0 or (os.path.exists(sigmafilename12))==0:
            print('{} does not exist!'.format(sigmafilename11))
            return None
        cli_11=np.loadtxt(sigmafilename11)
        cli_12=np.loadtxt(sigmafilename12)
        ci_11=np.loadtxt(sigmafilename11const)# for const
        basisnum=basis.gen_basisnum(imax)
        # print('imax=',imax,'shape of cli11',np.shape(cli_11))
        kbasis=np.empty((2,basisnum,knum,knum,knum),dtype=float)
        if rank==0:
            kbasis=basis.gen_kbasis_new(imax,knum)
        kbasis = np.ascontiguousarray(kbasis)
        comm.Bcast(kbasis, root=0)    
        # print('imax=',imax)
        # if ifchecksvd==1:
        #     if rank==0:
        #         basisind=basis.gen_basisindlist(imax)
        #         basisindnum=np.shape(basisind)[0]
        #         for ikb in np.arange(basisindnum):
        #             if np.sum(basisind[ikb])%2==0:
        #                 plt.plot(cli_11[:,ikb],label='({},{},{})'.format(basisind[ikb,0],basisind[ikb,1],basisind[ikb,2]))
        #         plt.title('cli11: U={} T={} order={} alpha={}'.format(U,T,order,alpha))
        #         plt.legend()
        #         plt.show()

        taulist=(np.arange(taunum+1))/taunum*beta
        ori_grid=(np.arange(nfreq*2)+0.5)/(nfreq*2)*beta
        Sig11tk_raw=basis.restore_tk(cli_11,ut,kbasis[0])
        Sig12tk_raw=basis.restore_tk(cli_12,ut,kbasis[1])
        interpolator_11 = interp1d(taulist, Sig11tk_raw, kind='cubic', axis=0, fill_value='extrapolate')
        interpolator_12 = interp1d(taulist, Sig12tk_raw, kind='cubic',  axis=0,fill_value='extrapolate')
        Sig11tk_full=interpolator_11(ori_grid)
        Sig12tk_full=interpolator_12(ori_grid)
        sig11const=basis.restore_k(ci_11,kbasis[0])
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
        if (os.path.exists(sigmafilename11))==0:
            print('{} does not exist!'.format(sigmafilename11))
            return None
        Sigiom_11=read_complex_numbers(sigmafilename11)[:,None,None,None]*np.ones((2*nfreq,knum,knum,knum))
        # Sigiom_11=np.loadtxt(sigmafilename11)[:,None,None,None]*np.zeros((2*nfreq,knum,knum,knum))
        Sigiom_12=np.zeros((2*nfreq,knum,knum,knum))
    Sigiom_22=U-Sigiom_11.conjugate()
    znew_1=z4D(beta,mu,Sigiom_11,knum,nfreq)
    znew_2=z4D(beta,mu,Sigiom_22,knum,nfreq)
    Gdress11_iom,Gdress12_iom=G_iterative(knum,znew_1,znew_2,Sigiom_12)
    Gdress22_iom=-Gdress11_iom.conjugate()
    return Sigiom_11,Sigiom_12,Sigiom_22,Gdress11_iom,Gdress12_iom,Gdress22_iom

#-------------magnetization----------
def mag_DMFT(U,T):
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

    magdmft=np.abs(particlenumber4D(G11_iom,beta)-particlenumber4D(G22_iom,beta))
    return magdmft

def mag_test(U,T,ifsearch=0,ifsave=0):
    '''
    This function is for testing. To check the self-energy stored in the form of coefficients can really be restored and give the same magnetization.
    '''
    foldernum=1
    beta=1/T
    maxorder=4
    orderstart=1
    # alpha_arr=np.array(([0.05,0.1]))
    # alpha_arr=np.array(([0.05,0.1,0.2,0.3,0.6,1.0]))
    # alpha_arr=np.array(([0.05,0.15,0.3,0.6,1.0]))
    # alpha_arr=np.array(([0.01,0.05,0.1,0.2,0.3]))
    # alpha_arr=np.array(([0.05,0.1,0.2,0.3,0.6]))
    # alpha_plt=np.array([0.05,0.1,0.15,0.3,0.6,1.0])
    alpha_plt=np.array([0.05,0.15,0.2,0.3,0.6,1.0])#0.1,0.1,
    # alpha_arraw=(np.arange(101))/100# to make sure only integer time of 0.03 exist.
    alpha_arraw = (np.arange(33)*3+3)/100
    # alpha_arraw = np.unique(np.concatenate(((np.arange(20)*3+3)/100, np.array([0.6,1.0]))))
    # print(alpha_arraw)
    alpha_arr=check_alpha(U,T,alpha_arraw,4,0)
    # print(alpha_arr)
    alphanum=np.shape(alpha_arr)[0]
    mag_arr=np.zeros((alphanum,maxorder+1))
    mag_arr2=np.zeros((alphanum,maxorder+1))

    magdmft=mag_DMFT(U,T)
    dircrit='../paperwriting/critmag/magdata/mag{}_{}.txt'.format(int(U),T)
    for order in np.arange(maxorder+1):
        for ialp,alpha in enumerate(alpha_arr):
            Sigiom_11,Sigiom_12,Sigiom_22,Gdress11_iom,Gdress12_iom,Gdress22_iom=get_sigma_and_G(U,T,order,alpha,foldernum)
            # znew_1=z4D(beta,mu,sigfinal11,knum,nfreq)
            # znew_2=z4D(beta,mu,sigfinal22,knum,nfreq)
            # Gdress11_iom,Gdress12_iom=G_iterative(knum,znew_1,znew_2,sigfinal12)
            # Gdress22_iom=-Gdress11_iom.conjugate()
            nnewloc11=particlenumber4D(Gdress11_iom,beta)
            nnewloc22=particlenumber4D(Gdress22_iom,beta)
            mag_arr[ialp,order]=nnewloc22-nnewloc11
    if ifsearch:
        foldernum2='_search'
        if U<8:
            searchorder=4
        else:
            searchorder=4
        alpha_arrsearch=check_alpha(U,T,alpha_arraw,searchorder,1)
        for order in np.arange(searchorder+1):
            for ialp,alpha in enumerate(alpha_arrsearch):
                Sigiom_11,Sigiom_12,Sigiom_22,Gdress11_iom,Gdress12_iom,Gdress22_iom=get_sigma_and_G(U,T,order,alpha,foldernum2)
                # znew_1=z4D(beta,mu,sigfinal11,knum,nfreq)
                # znew_2=z4D(beta,mu,sigfinal22,knum,nfreq)
                # Gdress11_iom,Gdress12_iom=G_iterative(knum,znew_1,znew_2,sigfinal12)
                # Gdress22_iom=-Gdress11_iom.conjugate()
                nnewloc11=particlenumber4D(Gdress11_iom,beta)
                nnewloc22=particlenumber4D(Gdress22_iom,beta)
                mag_arr2[ialp,order]=nnewloc22-nnewloc11
        alparr=np.concatenate((alpha_arr,alpha_arrsearch))
        magarr=np.concatenate((mag_arr,mag_arr2),axis=0)
        sorted_indices = np.argsort(alparr)
        allalphas=alparr[sorted_indices]
        allmags=magarr[sorted_indices,:]
        
        data=np.zeros((allalphas.size,7))#alpha, DMFT, order01234, 7 columns
        data[:,0]=allalphas
        data[:,1]=magdmft
        for ord in np.arange(maxorder+1):
            data[:,ord+2]=allmags[:,ord]
        if ifsave:
            np.savetxt(dircrit,data)
        if orderstart==3:
            plt.plot(allalphas, allmags[:,4]- allmags[:,3], marker='o', linestyle='-')
        else:
            plt.plot(allalphas, np.std(allmags[:,orderstart:],axis=1), marker='o', linestyle='-')

        plt.xlabel('alpha')
        plt.ylabel('standard deviation')
        plt.title('standard deviation U={} T={}'.format(U, T))
        plt.grid()
        plt.show()

        plt.axhline(y=magdmft, color='b', linestyle='--', linewidth=2,label='DMFT')
        plt.plot(allalphas, allmags[:,0], marker='o', linestyle='-',label='0th')
        plt.plot(allalphas, allmags[:,1], marker='^', linestyle='-',label='1st')
        plt.plot(allalphas, allmags[:,2], marker='s', linestyle='-',label='2nd')
        plt.plot(allalphas, allmags[:,3], marker='p', linestyle='-',label='3rd')
        plt.plot(allalphas, allmags[:,4], marker='h', linestyle='-',label='4th')
        plt.legend()
        plt.xlabel('alpha')
        plt.ylabel('magnetization')
        plt.title('mag vs alpha: U={} T={}'.format(U, T))
        plt.grid()
        plt.show()

        plt.figure(figsize=(4, 6))
        for ialp,alpha in enumerate(allalphas):
            if np.isin(alpha,alpha_plt):
                plt.plot(np.arange(maxorder+1), allmags[ialp,:], marker='o', linestyle='-',label='alpha={}'.format(alpha))
        plt.legend()
        plt.xlabel('order')
        plt.ylabel('magnetization')
        plt.title('mag vs Order: U={} T={}'.format(U, T))
        plt.grid()
        
        plt.show()

    else:
        # smoothening 4th order. for 0.03,0.06,....... up to 0.6
        smo_ind=np.where(alpha_arr<1)[0]# remember to control this upper bound
        indmax=smo_ind[-1]
        print(smo_ind)
        orimarr4=copy.deepcopy(mag_arr[:indmax+1,4])
        mag_arr[1:indmax+1,4]+=orimarr4[:-1]
        mag_arr[:indmax,4]+=orimarr4[1:]
        mag_arr[0,4]/=2
        mag_arr[indmax,4]/=2
        mag_arr[1:indmax,4]/=3
        mag_arr[0,4]=2*mag_arr[1,4]-mag_arr[2,4]
        
        # if orderstart==3:
        #     ax.plot(alpha_arr, mag_arr[:,4]- mag_arr[:,3], marker='o', linestyle='-')
        # else:
        #     ax.plot(alpha_arr, np.std(mag_arr[:,orderstart:],axis=1), marker='o', linestyle='-',label='T={}'.format(T))
        # ax.set_xlabel(r'$\alpha$')
        # ax.set_ylabel('standard deviation')
        # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        # plt.title('U={} T={}'.format(U, T))
        # plt.grid()
        # plt.savefig("../paperwriting/pert/alpha_{}_{}.png".format(U,T), dpi=1000)
        # plt.show()

        # plt.axhline(y=magdmft, color='b', linestyle='--', linewidth=2,label='DMFT')
        # plt.plot(alpha_arr, mag_arr[:,0], marker='o', linestyle='-',label='0th')
        # plt.plot(alpha_arr, mag_arr[:,1], marker='^', linestyle='-',label='1st')
        # plt.plot(alpha_arr, mag_arr[:,2], marker='s', linestyle='-',label='2nd')
        # plt.plot(alpha_arr, mag_arr[:,3], marker='p', linestyle='-',label='3rd')
        # plt.plot(alpha_arr, mag_arr[:,4], marker='h', linestyle='-',label='4th')
        # plt.legend()
        # plt.xlabel('alpha')
        # plt.ylabel('magnetization')
        # plt.title('mag vs alpha: U={} T={}'.format(U, T))
        # plt.grid()
        # plt.show()
        # data=np.zeros((alpha_arr.size,7))#alpha, DMFT, order01234, 7 columns
        # data[:,0]=alpha_arr
        # data[:,1]=magdmft
        # for ord in np.arange(maxorder+1):
        #     data[:,ord+2]=mag_arr[:,ord]
        # if ifsave:
        #     np.savetxt(dircrit,data)
        # plt.figure(figsize=(4, 6))
        # for ialp,alpha in enumerate(alpha_arr):
        #     if np.isin(alpha,alpha_plt):
        #         plt.plot(np.arange(maxorder+1), mag_arr[ialp,:], marker='o', linestyle='-',label='alpha={}'.format(alpha))
        # plt.legend()
        # plt.ylim(0,0.9)
        # plt.xlabel('order')
        # plt.ylabel('m')
        # plt.axhline(y=magdmft, color='b', linestyle='--', linewidth=2)
        # plt.title('U={} T={}'.format(U, T))
        # plt.tight_layout()
        # # plt.text(2.5, 0.45, 'U={} T={}'.format(U, T), fontsize=12, color="black")
        # plt.grid()
        # # plt.savefig("../paperwriting/pert/mag_{}_{}.png".format(U,T), dpi=1000)
        # plt.show()
    return 0

def mag_test2(U,Tmin,Tmax,order=4,ifplot=1):
    '''
    This function aims to roughly check how robust the MC result is. The idea is to check the magnetization dependence at specific alpha.
    '''
    Tlist=np.arange(np.round(Tmin*100),np.round(Tmax*100))/100
    print(Tlist,np.round(Tmin*100),np.round(Tmax*100))
    # alpha_arr=np.array(([0.01,0.05,0.1,0.2,0.3,0.6,1.0]))
    # alpha_arr=np.array(([0.05,0.1,0.2,0.3,0.6,1.0]))
    alpha_arr=np.array(([0.05,0.3,0.6,1.0]))
    mag_arr=np.zeros((alpha_arr.size,Tlist.size))
    for iT,T in enumerate(Tlist):
        beta=1/T
        for ialp,alpha in enumerate(alpha_arr):
        
            Sigiom_11,Sigiom_12,Sigiom_22,Gdress11_iom,Gdress12_iom,Gdress22_iom=get_sigma_and_G(U,T,order,alpha)
            nnewloc11=particlenumber4D(Gdress11_iom,beta)
            nnewloc22=particlenumber4D(Gdress22_iom,beta)
            mag_arr[ialp,iT]=nnewloc22-nnewloc11
    orimarr=copy.deepcopy(mag_arr)
    #smoothen
    # mag_arr[:,1:]+=orimarr[:,:-1]
    # mag_arr[:,:-1]+=orimarr[:,1:]
    # mag_arr[:,0]=orimarr[:,0]
    # mag_arr[:,-1]=orimarr[:,-1]
    # mag_arr[:,1:-1]/=3
    if ifplot==1:
        for ialp,alpha in enumerate(alpha_arr):
            plt.plot(Tlist, mag_arr[ialp,:],label='alpha={}'.format(alpha))    
        plt.legend()
        plt.xlabel('T')
        plt.ylabel('magnetization')
        plt.title('mag vs T: U={} order={}'.format(U,order))
        plt.grid()
        plt.show()
    return mag_arr


def E_test2(U,Tmin,Tmax,order=4,ifplot=1):
    Tlist=np.arange(np.round(Tmin*100),np.round(Tmax*100))/100
    # print(Tlist,np.round(Tmin*100),np.round(Tmax*100))
    # alpha_arr=np.arange(6)/5
    alpha_arr=np.array([0.05,0.1,0.2,0.3,0.6,1.0])
    E_arr=np.zeros((alpha_arr.size,Tlist.size))
    for iT,T in enumerate(Tlist):
        beta=1/T
        E,alphas=energy_test(U,T)
        Einterp=interp1d(alphas,E[:,order],axis=0, fill_value='extrapolate')
        E_arr[:,iT]=Einterp(alpha_arr)
    oriEarr=copy.deepcopy(E_arr)
    E_arr[:,1:]+=oriEarr[:,:-1]
    E_arr[:,:-1]+=oriEarr[:,1:]
    E_arr[:,0]=oriEarr[:,0]
    E_arr[:,-1]=oriEarr[:,-1]
    E_arr[:,1:-1]/=3
    if ifplot==1:
        for ialp,alpha in enumerate(alpha_arr):
            plt.plot(Tlist, E_arr[ialp,:],label='alpha={}'.format(alpha))    
        plt.legend()
        plt.xlabel('T')
        plt.ylabel('E')
        plt.title('E vs T: U={} order={}'.format(U,order))
        plt.grid()
        plt.show()
    return E_arr



# ------------total energy-----------
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
    res=get_sigma_and_G(U,T,order,alpha)
    if res==None:
    # if isinstance(res, tuple) and all(x is None for x in res):
        return (None,None)
    Sigiom_11,Sigiom_12,Sigiom_22,Gdress11_iom,Gdress12_iom,Gdress22_iom=res
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
    Edisp=(H0_G+TrSigmaG/2)# we have to sites, and we're looking for E per site.
    # check impurity total energy.

    return Edisp,TrSigmaG/2

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
    EDMFT=(H0_G+TrSigmaG/2)
    return EDMFT,TrSigmaG/2

def energy_test(U,T,ifplot=0,magopt=0):
    beta=1/T
    maxorder=4

    # if magopt==1:# AFM
    #     alpha_arr=np.array(([0.2,0.3,0.6,1.0]))
    # elif magopt==2:#PM
    #     alpha_arr=np.array(([0.05,0.1,0.2,0.6]))
    # elif magopt==0:
    #     alpha_arr=np.array(([0.05,0.1,0.2,0.3,0.6,1.0]))
    alpha_plt=np.array([0.05,0.15,0.2,0.3,0.6,1.0])
    alpha_arrraw=np.arange(101)/100
    alpha_arr=check_alpha(U,T,alpha_arrraw,4,0)
    # print(alpha_arr)
    alphanum=np.shape(alpha_arr)[0]
    E_arr=np.zeros((alphanum,maxorder+1))
    dboc_arr=np.zeros((alphanum,maxorder+1))
    EDMFT_arr=np.zeros((alphanum))
    dbocDMFT_arr=np.zeros((alphanum))
    for ialp,alpha in enumerate(alpha_arr):
        EDMFT_arr[ialp],dbocDMFT_arr[ialp]=total_energy_DMFT(U,T)
        for order in np.arange(maxorder+1):
        
            E_arr[ialp,order],dboc_arr[ialp,order]=total_energy(U,T,order,alpha)
    # print(E_arrraw)
    # validrows=get_valid_rows_indices(E_arrraw)
    # E_arr=E_arrraw[validrows]
    # dboc_arr=dboc_arrraw[validrows]
    # alpha_arr=alpha_arrraw[validrows]
    if ifplot==1:
        
        plt.axhline(y=EDMFT_arr[0],linestyle='--', label='DMFT')
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

        plt.figure(figsize=(4.5, 6))
        plt.axhline(y=EDMFT_arr[0],linestyle='--', color='b')# label='DMFT'
        for ialp,alpha in enumerate(alpha_arr):
            if np.isin(alpha,alpha_plt):
                plt.plot(np.arange(maxorder+1), E_arr[ialp,:], marker='o', linestyle='-',label='alpha={}'.format(alpha))
        plt.legend()
        plt.xlabel('order')
        plt.ylabel('E')
        plt.ylim(-5.1,-4.15)
        plt.grid()
        # plt.title('Total energy vs Order: U={} T={}'.format(U, T))
        plt.title('U={} T={}'.format(U, T))
        plt.tight_layout()
        plt.savefig("../paperwriting/pert/E_{}_{}.png".format(U,T), dpi=1000)
        # plt.show()
    return E_arr,alpha_arr


def energy_with_errbar(U,order,TN=0):
    '''
    
    '''
    T_bound=np.array(((4.0,0.1,0.7),(5.,0.2,0.31),
        (8.,0.2,0.58),(10.,0.25,0.63),(12.,0.2,0.69),(14.,0.26,0.5)))
    Ufound=0
    for list in T_bound:
        if list[0]==U:
            listT=list
            Ufound=1
            break
    if Ufound==0:
        print('U not found!')
        return 0
    alpha_arr=np.arange(21)/20
    # alpha_arr=np.array(([0.05,0.1,0.2,0.3,0.6,1.0]))
    alphanum=np.shape(alpha_arr)[0]



    T_arr=np.arange(int(listT[1]*100),int(listT[2]*100))/100
    Emax=np.zeros_like(T_arr)
    Emin=np.zeros_like(T_arr)
    Eimp=np.zeros_like(T_arr)
    allE=np.zeros((alphanum,T_arr.size))
    for iT,T in enumerate(T_arr):
        print(T)
        E,alphaarr=energy_test(U,T)
        E_array=E[:,order]
        Einterp=interp1d(alphaarr,E_array,kind='linear', fill_value='extrapolate')
        allE[:,iT]=Einterp(alpha_arr)
        Emax[iT]=max(Einterp(alpha_arr))#[10:]
        Emin[iT]=min(Einterp(alpha_arr))#[10:]
        # if T<TN:
        #     Emin[iT]=min(E_array[3:])
        Eimp[iT]=total_energy_DMFT(U,T)[0]
        E_ave=(Emax+Emin)/2
        E_err=(Emax-Emin)/2


    Cv=-(allE[:,:-1]-allE[:,1:])/0.01
    T_arr2=(T_arr[:-1]+T_arr[1:])/2
    Cvmax=np.zeros_like(T_arr2)
    Cvmin=np.zeros_like(T_arr2)
    for iT,T in enumerate(T_arr2):
        Cvmax[iT]=max(Cv[10:,iT])
        Cvmin[iT]=min(Cv[10:,iT])
    Cv_ave=(Cvmax+Cvmin)/2
    Cv_err=(Cvmax-Cvmin)/2
    plt.errorbar(T_arr, E_ave, yerr=E_err, fmt='-o', capsize=5,color='r',label='4th order')
    plt.plot(T_arr,Eimp,color='b', linestyle='--',label='DMFT')
    plt.ylabel('Total Energy')
    plt.xlabel('Temperature')
    plt.title('Temperature dependence of total energy: U={}'.format(U))
    plt.legend()
    plt.show()
    h=0.01
    dE_pert_dT = -(-E_ave[:-4] + 8*E_ave[1:-3] - 8*E_ave[3:-1] + E_ave[4:]) / (12*h)
    dE_DMFT_dT = -(-Eimp[:-4] + 8*Eimp[1:-3] - 8*Eimp[3:-1] + Eimp[4:]) / (12*h)
    T_mid = T_arr[2:-2]
    plt.errorbar(T_arr2, Cv_ave, yerr=Cv_err, fmt='-o', capsize=5,color='r',label='Cv=dE/dT')
    plt.plot(T_mid, dE_pert_dT, label='dE_pert/dT (4th order)')
    plt.plot(T_mid, dE_DMFT_dT, label='dE_DMFT/dT ')
    plt.legend()
    plt.show()
    # plt.errorbar(T_arr2, Cv_ave, yerr=Cv_err, fmt='-o', capsize=5,color='r',label='Cv=dE/dT')
    # plt.plot((T_arr[:-1]+T_arr[1:])/2,(-E_ave[:-1]+E_ave[1:])/0.01,label='dE_pert/dT')
    # plt.plot((T_arr[:-1]+T_arr[1:])/2,(-Eimp[:-1]+Eimp[1:])/0.01,label='dE_DMFT/dT')
    # plt.ylabel('dE/dT')
    # plt.xlabel('Temperature')
    # plt.title('Temperature dependence of Specific heat: U={} order={}'.format(U,order))
    # plt.legend()
    # plt.show()    
    return 0

def energy_fastconv(U,ifplot=0,ifread=1):
    T_bound=np.array(((4.0,0.1,0.4),(8.,0.2,0.58),(10.,0.25,0.63),(12.,0.2,0.69),(14.,0.26,0.5)))
    Ufound=0
    for list in T_bound:
        if list[0]==U:
            listT=list
            Ufound=1
            break
    if Ufound==0:
        print('U not found!')
        return 0
    maxord=4
    alpha_arr=np.arange(101)/100
    alphanum=np.shape(alpha_arr)[0]
    # T_arr=np.arange(int(listT[1]*100),int(listT[2]*100))/100
    T_arr=get_Tlist(U)
    Eimp=np.zeros_like(T_arr)
    Eopt=np.zeros_like(T_arr)
    Eerr=np.zeros_like(T_arr)
    allE=np.zeros((alphanum,maxord+1,T_arr.size))
    orderst=0
    orderend=4
    dirE='../paperwriting/Cv/U{}calc.txt'.format(int(U))
    if (os.path.exists(dirE))==0 or ifread==0:

        for iT,T in enumerate(T_arr):
            print('T=',T)
            E_array,alphaarr=energy_test(U,T,0)# Earray is alpha,order
            # print(E_array,alpha_arr)
            Einterp=interp1d(alphaarr,E_array,axis=0,kind='linear', fill_value='extrapolate')
            allE[:,:,iT]=Einterp(alpha_arr)#alpha,order,T
        # smoothen, if needed.
        oriEarr=copy.deepcopy(allE)
        allE[:,:,1:]+=oriEarr[:,:,:-1]
        allE[:,:,:-1]+=oriEarr[:,:,1:]
        allE[:,:,0]=oriEarr[:,:,0]
        allE[:,:,-1]=oriEarr[:,:,-1]
        allE[:,:,1:-1]/=3
        for iT,T in enumerate(T_arr):
            var=np.std(allE[:,orderst:,iT],axis=1)
            if U==4:
                var=np.std(allE[:,orderst:orderend,iT],axis=1)
            minind=np.argmin(var)

            Eopt[iT]=np.average(allE[minind,orderst:,iT])# this can be tuned.
            if U==4:
                Eopt[iT]=np.average(allE[minind,orderst:orderend,iT])
            Eerr[iT]=var[minind]
            Eimp[iT]=total_energy_DMFT(U,T)[0]
            print('T={}, optalpha~{} var~{}'.format(T,alpha_arr[minind],Eerr[iT]))

        
        data=np.zeros((T_arr.size,4))
        data[:,0]=T_arr
        data[:,1]=Eimp
        data[:,2]=Eopt
        data[:,3]=Eerr
        np.savetxt(dirE,data)
    else:
        data=np.loadtxt(dirE)
        T_arr=data[:,0]
        Eimp=data[:,1]
        Eopt=data[:,2] 
        Eerr=data[:,3]       
    # to suppress the noice in Cv we have to process our signal.
    Tuni=np.arange(T_arr[0],T_arr[-1],0.01)
    E_interp = interp1d(T_arr, Eopt, kind='linear')
    E_uni=E_interp(Tuni)
    E_smooth=E_uni
    E_smooth = savgol_filter(E_uni, window_length=11, polyorder=3)
    E_smooth[22:] = savgol_filter(E_smooth[22:], window_length=7, polyorder=1)
    E_smooth = savgol_filter(E_smooth, window_length=5, polyorder=3)
    plt.plot(E_smooth)
    plt.plot(E_uni)
    plt.show()
    Eimp_interp = interp1d(T_arr, Eimp, kind='linear')
    Eimp_uni=Eimp_interp(Tuni)
    Eimp_smooth = Eimp_uni#savgol_filter(Eimp_uni, window_length=5, polyorder=3)
    # for Cv we cut the T range in 2 parts. the first part use high-order differenciation.
    h=0.01
    dE_pert_dT = -(-E_smooth[:-4] + 8*E_smooth[1:-3] - 8*E_smooth[3:-1] + E_smooth[4:]) / (12*h)
    dE_DMFT_dT = -(-Eimp_smooth[:-4] + 8*Eimp_smooth[1:-3] - 8*Eimp_smooth[3:-1] + Eimp_smooth[4:]) / (12*h)
    T_mid = T_arr[2:-2]
    Tuni_mid = Tuni[2:-2]
    # Tarrsmall = T_mid[np.where(T_mid <0.55)[0]]
    # #low order, for highT
    # Thigh=(T_arr[:-1]+T_arr[1:])/2
    # Cvhigh=(-Eopt[:-1]+Eopt[1:])/(-T_arr[:-1]+T_arr[1:])
    # CvDMFThigh=(-Eimp[:-1]+Eimp[1:])/(-T_arr[:-1]+T_arr[1:])

    # Tarrlarge = Thigh[np.where(Thigh >=0.55)[0]]
    # Cvsmall=dE_pert_dT[np.where(T_mid <0.55)[0]]
    # CvDMFTsmall=dE_DMFT_dT[np.where(T_mid <0.55)[0]]
    # Cvlarge=Cvhigh[np.where(Thigh >=0.55)[0]]
    # CvDMFTlarge=CvDMFThigh[np.where(Thigh >=0.55)[0]]

    # T_arrnew=np.concatenate((Tarrsmall,Tarrlarge))
    # Cv_arrnew=np.concatenate((Cvsmall,Cvlarge))
    # CvDMFT_arrnew=np.concatenate((CvDMFTsmall,CvDMFTlarge))
    if ifplot==1:



        fig, ax = plt.subplots()

        if U==8 or U==4 or U==12:
            dir='../paperwriting/Cv/U{}.txt'.format(int(U))
            data=np.loadtxt(dir, delimiter=',')
            ax.plot(data[:,0],data[:,1],label='QMC')
        ax.plot(Tuni_mid[2:],dE_pert_dT[2:],label='VDMC(AF)')
        ax.plot(Tuni_mid[2:],dE_DMFT_dT[2:],label='DMFT')
        # ax.axvspan(0.29, 0.355, color='red', alpha=0.2)
        ax.axvline(x=0.355, color='orange', linestyle='--')
        ax.axvline(x=0.43, color='green', linestyle='--')
        # ax.plot(T_arrnew,Cv_arrnew,label='VP(AF)')
        # ax.plot(T_arrnew,CvDMFT_arrnew,label='DMFT')
        inset_ax = inset_axes(ax, width="50%", height="50%", loc='upper right') 
        inset_ax.errorbar(T_arr, Eopt, yerr=Eerr, fmt='-o', markersize=3,capsize=1,color='r',label='VDMC(AF)')
        inset_ax.plot(T_arr,Eimp,color='b', linestyle='--',label='DMFT')
        # inset_ax.axvspan(0.29, 0.355, color='red', alpha=0.2)
        inset_ax.axvline(x=0.355, color='orange', linestyle='--')
        inset_ax.axvline(x=0.43, color='green', linestyle='--')
        inset_ax.set_ylabel('E')
        inset_ax.set_xlabel('T')
        # inset_ax.set_xlim(0.,1.5)
        # plt.title('Temperature dependence of total energy: U={}'.format(U))
        inset_ax.legend()
        inset_ax.set_xscale('log')


        ax.set_ylabel('Specific Heat')
        ax.set_xlabel('T')
        # ax.title('Temperature dependence of Specific heat: U={}'.format(U))
        ax.legend(loc='lower right')
        # plt.ylim(0,1.5)
        # plt.xlim(0.25,0.5)
        ax.set_xscale('log')
        plt.savefig("Fig3.png", dpi=1000)
        plt.show()    
    # return Cv_arrnew,CvDMFT_arrnew,T_arrnew
    return dE_pert_dT,dE_DMFT_dT,T_mid
    


def energy_with_errbarallorders(U,ifplot=1):
    T_bound=np.array(((3.0,0.08,0.14),(5.,0.2,0.31),
        (8.,0.2,0.52),(10.,0.25,0.63),(12.,0.24,0.6),(14.,0.26,0.5)))
    Ufound=0
    for list in T_bound:
        if list[0]==U:
            listT=list
            Ufound=1
            break
    if Ufound==0:
        print('U not found!')
        return 0
    maxorder=4
    # alpha_arr=np.array(([0.05,0.1,0.2,0.3,0.6,1.0]))
    alpha_arr=np.arange(21)/20
    alphanum=np.shape(alpha_arr)[0]
    T_arr=np.arange(int(listT[1]*100),int(listT[2]*100))/100
    Tnum=np.shape(T_arr)[0]
    Emax=np.zeros((maxorder+1,Tnum))
    Emin=np.zeros((maxorder+1,Tnum))
    Eimp=np.zeros_like(T_arr)
    allE=np.zeros((alphanum,maxorder+1,T_arr.size))
    for iT,T in enumerate(T_arr):
        Eimp[iT]=total_energy_DMFT(U,T)[0]

        # for order in np.arange(maxorder+1):
        E_array,alphas=energy_test(U,T)
        Einterp=interp1d(alphas,E_array,kind='linear',axis=0, fill_value='extrapolate')
        allE[:,:,iT]=Einterp(alpha_arr)#E_array# alpha,order
        Emax[:,iT]=np.max(Einterp(alpha_arr),axis=0)#[4:]
        Emin[:,iT]=np.min(Einterp(alpha_arr),axis=0)#[4:]
            # if T<TN:
            #     Emin[iT]=min(E_array[3:])
    E_ave=(Emax+Emin)/2
    E_err=(Emax-Emin)/2
    Cv=-(allE[:,:,:-1]-allE[:,:,1:])/0.01#3 dimensions: alpha, order,T
    T_arr2=(T_arr[:-1]+T_arr[1:])/2
    Cvmax=np.zeros((maxorder+1,Tnum-1))
    Cvmin=np.zeros((maxorder+1,Tnum-1))
    for iT,T in enumerate(T_arr2):
        Cvmax[:,iT]=np.max(Cv[10:,:,iT],axis=0)
        Cvmin[:,iT]=np.min(Cv[10:,:,iT],axis=0)
    Cv_ave=(Cvmax+Cvmin)/2
    Cv_err=(Cvmax-Cvmin)/2
    Cv_DMFT=(-Eimp[:-1]+Eimp[1:])/0.01
    if ifplot:
        for order in np.arange(maxorder+1):
            plt.errorbar(T_arr, E_ave[order], yerr=E_err[order], fmt='-o', capsize=5,label='order={}'.format(order))
        plt.plot(T_arr,Eimp, linestyle='--',label='DMFT')
        plt.ylabel('Total Energy')
        plt.xlabel('Temperature')
        plt.title('Temperature dependence of total energy: U={}'.format(U))
        plt.legend()
        plt.show()

    
        # for order in np.arange(maxorder+1):
        for order in np.array([3,4]):
            plt.errorbar(T_arr2, Cv_ave[order], yerr=Cv_err[order], fmt='-o', capsize=5,label='Cv=dE/dT order={}'.format(order))
        plt.plot((T_arr[:-1]+T_arr[1:])/2,(-Eimp[:-1]+Eimp[1:])/0.01,label='dE_DMFT/dT')
        plt.ylabel('dE/dT')
        plt.xlabel('Temperature')
        plt.title('Temperature dependence of Specific heat: U={}'.format(U))
        plt.legend()
        # plt.ylim(0,1.5)
        # plt.xlim(0.25,0.5)
        # plt.xscale('log')
        plt.show()    
    return Cv,Cv_DMFT,T_arr,Cv_ave
        
def entropy_from_back_integration(U,order,TN=0):
    '''
    Since DMFT is good at high T, it a good idea to start at high T and integrate the S.
    '''
    Cv_arrnew,CvDMFT_arrnew,T_arrnew=energy_fastconv(U)
    T_arr=get_Tlist(U)
    S_int=np.zeros_like(T_arrnew)
    S_intDMFT=np.zeros_like(T_arrnew)
    # S at a high T point using (E-F)/T
    # FDMFThighT=free_energy_test(U,T_arr[-1])
    Fimp,FDMFThighT, Fdisp=gen_free_energy_files(U,T_arr[-1])
    EDMFThighT=total_energy_DMFT(U,T_arr[-1])[0]
    SDMFThighT=(EDMFThighT-FDMFThighT)/T_arr[-1]

    S_int[-1]=SDMFThighT
    S_intDMFT[-1]=SDMFThighT
    Tarr_final=np.zeros_like(T_arrnew)
    Tarr_final[:-1]=(T_arrnew[1:]+T_arrnew[:-1])/2
    Tarr_final[-1]=T_arr[-1]


    Tarrp=Tarr_final[:-1]# points have to be calculated.
    # Cvorder=Cv[:,order,:]     #3 dimensions: alpha, order,T
    # print(T_arr)
    # print(T_arrnew)
    Tsize=T_arrnew.size
    for iT,T in enumerate(Tarrp[::-1]):
        S_int[Tsize-iT-2]=S_int[Tsize-iT-1]-Cv_arrnew[Tsize-iT-2]/T*(T_arrnew[Tsize-1-iT]-T_arrnew[Tsize-2-iT])
        # S_intfromave[Tsize-iT-2]=S_intfromave[Tsize-iT-1]-Cv_ave[order,Tsize-iT-2]/T*0.01
        S_intDMFT[Tsize-iT-2]=S_intDMFT[Tsize-iT-1]-CvDMFT_arrnew[Tsize-iT-2]/T*(T_arrnew[Tsize-1-iT]-T_arrnew[Tsize-2-iT])
    if U==8 or U==4 or U==12:
        dir='../paperwriting/Entropy/S{}.txt'.format(int(U))
        data=np.loadtxt(dir, delimiter=',')
        plt.plot(data[:,0],data[:,1],label='QMC')
    # plt.errorbar(T_arr, S_intave, yerr=S_interr, fmt='-o', capsize=5,label='order={}'.format(order))
    plt.plot(Tarr_final,S_int,label='PERT int')
    plt.plot(Tarr_final,S_intDMFT,label='DMFT int')
    plt.legend()
    plt.title('Entropy: U={}'.format(U))
    # plt.ylim(0,1)
    # plt.xlim(0.1,1)
    # plt.xscale('log')
    plt.show()
    return 0

#------------- free energy part. ----------
#Since free energy diagrammatic resummation is not good, don't use them.
def free_energy_test(U,T,ifplot=0):
    beta=1/T
    maxorder=4
    # alpha_arr=np.array(([0.01,0.05,0.1,0.15,0.2,0.3,0.4,0.6,0.8,1.0]))
    energy_dir='./energydata/F_{}_{}.dat'.format(U,T)
    data=np.loadtxt(energy_dir)
    alpha_arr=data[:,0]
    # alphanum=np.shape(alpha_arr)[0]
    # F_arr=np.zeros((alphanum,maxorder+1))
    # for order in np.arange(maxorder+1):
    #     for ialp,alpha in enumerate(alpha_arr):
    #         F_arr[ialp,order]=data[ialp,order+1]

    if ifplot==1:
        plt.plot(alpha_arr, data[:,2],  linestyle='--',label='DMFT')
        plt.plot(alpha_arr, data[:,3], marker='o', linestyle='-',label='0th')
        plt.plot(alpha_arr, data[:,4], marker='^', linestyle='-',label='1st')
        plt.plot(alpha_arr, data[:,5], marker='s', linestyle='-',label='2nd')
        plt.plot(alpha_arr, data[:,6], marker='p', linestyle='-',label='3rd')
        plt.plot(alpha_arr, data[:,7], marker='h', linestyle='-',label='4th')
        plt.legend()
        plt.title('Free energy vs Alpha: U={} T={}'.format(U, T))
        plt.xlabel('alpha')
        plt.ylabel('Free energy')
        plt.show()
    return data

def gen_free_energy_files(U,T):
    '''
    the structure of Free energy files:
    alpha   Fimp     FDMFT   F_(0)  F_(1) F_(2) F_(3) F_(4)
    
    '''
    mu=U/2
    beta=1/T
    knum=10
    nfreq=500
    maxorder=1
    alpha_arr=np.array(([0.05]))#,0.1,0.2,0.3,0.6,1.0
    alphanum=np.shape(alpha_arr)[0]
    F_arr=np.zeros((alphanum,maxorder+4))
    for order in np.arange(maxorder+1):
        for ialp,alpha in enumerate(alpha_arr):
            if rank==0:
                print('U={} T={} order={} alpha={}'.format(U,T,order,alpha))
            # #for testing
            # alpha=1.0
            # order=4
            Sigiom_11,Sigiom_12,Sigiom_22,Gdress11_iom,Gdress12_iom,Gdress22_iom=get_sigma_and_G(U,T,order,alpha)
            Fimp,F_DMFT, Fdisp=energy.PertFreeEnergy(Sigiom_11,Sigiom_22,Sigiom_12,U,T,order,0)
            F_arr[ialp,order+3]=Fdisp
            F_arr[ialp,0]=alpha
            F_arr[ialp,1]=Fimp
            F_arr[ialp,2]=F_DMFT
    filename='./energydata/F_{}_{}.dat'.format(U,T)
    np.savetxt(filename,F_arr)
    return Fimp,F_DMFT, Fdisp


def gen_all_free_energies(U):
    T_bound=np.array(((8.,0.25,0.58),(10.,0.31,0.63),(14.,0.26,0.5)))
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
    for iT,T in enumerate(T_arr):
        dir='../files_boldc/{}_{}/ctqmc.log'.format(U,T)
        dirF='./energydata/F_{}_{}.dat'.format(U,T)
        if (os.path.exists(dir))==0:
            if rank==0:
                print('ctqmc.log not found!')
            
        elif (os.path.exists(dirF)):
            if rank==0:
                print('already have {}. skipped!'.format(dirF))   
        else:   
            gen_free_energy_files(U,T)      

    return 0

def free_energy_with_errbar(U,order,TN=0):
    '''
    
    '''
    T_bound=np.array(((8.,0.25,0.58),(10.,0.31,0.6),(14.,0.26,0.5)))
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
    F_DMFT=np.zeros_like(T_arr)
    for iT,T in enumerate(T_arr):
        Farr=free_energy_test(U,T)
        F_array=Farr[:,order+3]
        Fmax[iT]=max(F_array)#[:5]
        Fmin[iT]=min(F_array)#[:5]
        F_DMFT[iT]=Farr[0,2]
        # if T<TN:
        #     Fmax[iT]=max(F_array[5:])#
        #     Fmin[iT]=min(F_array[5:])#
        # Eimp[iT]=total_energy_DMFT(U,T)
    F_ave=(Fmax+Fmin)/2
    F_err=(Fmax-Fmin)/2
    plt.errorbar(T_arr, F_ave, yerr=F_err, fmt='-o', capsize=5,color='r',label='3rd order')
    plt.plot(T_arr,F_DMFT,color='b', linestyle='--',label='DMFT')
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


def S_with_errbar(U,order,TN=0):
    '''
    
    '''
    T_bound=np.array(((8.,0.25,0.57),(10.,0.31,0.63),(12.,0.31,0.7),(14.,0.26,0.5)))
    Ufound=0
    for list in T_bound:
        if list[0]==U:
            listT=list
            Ufound=1
            break
    if Ufound==0:
        print('U not found!')
        return 0
    alpha_arr=np.array(([0.05,0.1,0.2,0.3,0.6,1.0]))
    alphanum=np.shape(alpha_arr)[0]
    T_arr=np.arange(int(listT[1]*100),int(listT[2]*100))/100
    Smax=np.zeros_like(T_arr)
    Smin=np.zeros_like(T_arr)
    S_array=np.zeros_like(T_arr)
    Emin=np.zeros_like(T_arr)
    Eimp=np.zeros_like(T_arr)
    F_DMFT=np.zeros_like(T_arr)
    S_DMFT=np.zeros_like(T_arr)
    allF=np.zeros((alphanum,T_arr.size))
    for iT,T in enumerate(T_arr):
        Farr=free_energy_test(U,T)
        
        F_array=Farr[:,order+3]
        # allF[:,iT]=F_array
        F_DMFT[iT]=Farr[0,2]

        # E_array=energy_test(U,T)[:,order]

        Eimp[iT]=total_energy_DMFT(U,T)[0]

        # S_array=-(F_array-E_array)/T
        S_DMFT[iT]=-(F_DMFT[iT]-Eimp[iT])/T
        # Smax[iT]=max(S_array[3:])#
        # Smin[iT]=min(S_array[3:])#

        # if T<TN:
        #     Smax[iT]=max(S_array[3:])#
        #     Smin[iT]=min(S_array[3:])#            
    # S_ave=(Smax+Smin)/2
    # S_err=(Smax-Smin)/2


    # SdFdT=(allF[:,:-1]-allF[:,1:])/0.01
    T_arr2=(T_arr[:-1]+T_arr[1:])/2
    # SdFdTmax=np.zeros_like(T_arr2)
    # SdFdTmin=np.zeros_like(T_arr2)
    # for iT,T in enumerate(T_arr2):
    #     SdFdTmax[iT]=max(SdFdT[3:,iT])
    #     SdFdTmin[iT]=min(SdFdT[3:,iT])
    # SdFdT_ave=(SdFdTmax+SdFdTmin)/2
    # SdFdT_err=(SdFdTmax-SdFdTmin)/2




    # plt.errorbar(T_arr, S_ave, yerr=S_err, fmt='-o', capsize=5,color='r',label='(E-F)/T')
    # plt.errorbar(T_arr2, SdFdT_ave, yerr=SdFdT_err, fmt='-o', capsize=5,color='g',label='-dF/dT')
    plt.plot(T_arr,S_DMFT,color='r', linestyle='-',label='(E-F)/T DMFT')
    # plt.plot(T_arr2,(F_DMFT[:-1]-F_DMFT[1:])/0.01,color='g', linestyle='--',label='-dF/dT DMFT')
    plt.ylabel('entropy')
    plt.xlabel('Temperature')
    plt.title('Temperature dependence of entropy: U={}'.format(U))
    plt.legend()
    plt.xlim(0,1)
    plt.ylim(0.3,1)
    plt.show()
    # plt.plot(T_arr[2:-2],(-S_ave[4:]+8*S_ave[3:-1]-8*S_ave[1:-3]+S_ave[:-4])/0.12,label='dS_pert/dT high order')
    # plt.plot(T_arr[2:-2],(-S_DMFT[4:]+8*S_DMFT[3:-1]-8*S_DMFT[1:-3]+S_DMFT[:-4])/0.12,label='dS_DMFT/dT high order')

    # plt.plot((T_arr[:-1]+T_arr[1:])/2,(-S_ave[:-1]+S_ave[1:])/0.01,label='dS_pert/dT')
    # plt.plot((T_arr[:-1]+T_arr[1:])/2,(-S_DMFT[:-1]+S_DMFT[1:])/0.01,label='dS_DMFT/dT')
    # plt.ylabel('dS/dT')
    # plt.xlabel('Temperature')
    # plt.legend()
    # plt.show()    
    return 0

    



if __name__ == "__main__":
    sizefont=14# default:12
    plt.rc('font', size=sizefont) 
    plt.rc('axes', titlesize=sizefont) 
    plt.rc('axes', labelsize=sizefont) 
    plt.rc('xtick', labelsize=sizefont)
    plt.rc('ytick', labelsize=sizefont)
    plt.rc('legend', fontsize=12)
    # get_sigma_and_G(8.,0.25,4,0.05,1,1)
    # plt.figure(figsize=(6, 4))
    # fig, ax = plt.subplots()
    # mag_test(8.,0.285,0,0)
    # mag_test(8.,0.38,0,0)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("../paperwriting/pert/alpha_{}.png".format(8), dpi=1000)
    # plt.show()
    # mag_test2(8.,0.2,0.3,4)
    # E_test2(8.,0.2,0.55,4)
    # energy_test(8.,0.25,1,1)
    # energy_with_errbar(8.,4.0.1)
    energy_fastconv(8.0,1,1)
    # energy_with_errbarallorders(8.)
    # entropy_from_back_integration(8.,4)