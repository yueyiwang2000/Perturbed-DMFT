#!/usr/bin/env python
import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess
# here we assume the sigma has 5 columns.
"""
This file contains useful functions to examine various output files.
"""
# sizefont=20# default:12
# plt.rc('font', size=sizefont) 
# plt.rc('axes', titlesize=sizefont) 
# plt.rc('axes', labelsize=sizefont) 
# plt.rc('xtick', labelsize=sizefont) 
# plt.rc('ytick', labelsize=sizefont) 
def plot_Tn():
    Ulist=np.array([3,4,5,6,7,8,9,10,11,12,13,14,15])#
    Tlist=np.array([0.105,0.18,0.25,0.34,0.4,0.45,0.475,0.49,0.48,0.45,0.43,0.395,0.375])
    Tpertlist=np.array([0.105,0.18,0.25,0.32,0.36,0.38,0.4,0.39,0.37,0.35,0.33,0.32,0.31])
    plt.plot(Ulist,Tlist,'o-',label='Neel Temperature')
    plt.plot(Ulist,Tpertlist,':',color='red',label='Expected Corrected Neel Temperature')
    plt.xlabel('U')
    plt.ylabel('T')
    plt.legend()
    plt.show()
    return 0
    

def plot_sigma(filename,index):
    sigma=np.loadtxt(filename)
    omega=sigma[:,0]
    # for i in np.arange(np.size(sigma[1,:])-1)+1:
    #     plt.plot(omega,sigma[:,i],label='{}th column in {}th file'.format(i,index))
    plt.plot(omega,sigma[:,1],label='real1 in {}th file'.format(index))
    # plt.plot(omega,sigma[:,3],label='real2 in {}th file'.format(index))
    # plt.plot(omega,np.zeros_like(omega),label='ZERO')
    plt.legend()
    # plt.xlim((0,omega[-1]/10))
    # plt.ylim((-0.2,0.2))

def single_sigma(filename):
    sigma=np.loadtxt(filename)
    omega=sigma[:,0]
    plt.plot(omega,sigma[:,1],label='real1 ')
    plt.plot(omega,sigma[:,3],label='real2 ')
    plt.plot(omega,sigma[:,2],label='imag1 ')
    plt.plot(omega,sigma[:,4],label='imag2 ')
        # plt.plot(omega,np.zeros_like(omega),label='ZERO')
    plt.legend()
    plt.grid()
    plt.show()
    # plt.xlim((0,omega[-1]/10))
    # plt.ylim((-0.2,0.2))

#single mode
def single_mode(num):
    ilist=np.arange(num)
    for i in ilist:
        filename='./'
        # filename='./files/{}_{}/Delta.OCA.{}'.format(U,T,int(i+1))
        # filename='./files/{}_{}/Sig.pert.dat.{}'.format(U,T,int(i+1))
        filename='./files_pert_boldc/{}_{}/sig12.dat.{}'.format(U,T,int(i+1))
        plot_sigma(filename,i)
        plt.legend()
        plt.title(filename)
        plt.show()

def stable_test(num):
    ilist=np.arange(num)
    # ilist=np.array([0,2])
    for i in ilist:
        # filename='./files/{}_{}/Sig.OCA.{}'.format(U,T,int(i+1))
        # filename='./files/{}_{}/Delta.OCA.{}'.format(U,T,int(i+1))
        filename='./files/{}_{}/Sig.out.{}'.format(U,T,int(i))
        sigma=np.loadtxt(filename)
        omega=sigma[:,0]
        plt.plot(omega,sigma[:,1],label='perturbed DMFT self-energy {}'.format(i),color='red')

        filename='./files/{}_{}/ori_Sig.out.{}'.format(U,T,int(i))
        sigma=np.loadtxt(filename)
        omega=sigma[:,0]
        plt.plot(omega,sigma[:,1],label='original DMFT self-energy {}'.format(i),color='blue')
    filename='./trial_sigma/{}_{}.dat'.format(U,T)
    sigma=np.loadtxt(filename)
    omega=sigma[:,0]
    plt.plot(omega,sigma[:,1],label='original sigma we start with',color='green')
    plt.legend()
    plt.show()

def compare_G(ind):
    freq_num=500
    filenameloc='../files_variational/{}_{}_{}/Gfloc.OCA.{}'.format(B,U,T,ind)
    filenameimp='../files_variational/{}_{}_{}/Gf.OCA.{}'.format(B,U,T,ind)
    Gloc=np.loadtxt(filenameloc)
    Gimp=np.loadtxt(filenameimp)
    omega=Gloc[:freq_num,0]
    G_loc=Gloc[:freq_num,3]+Gloc[:freq_num,4]*1j
    G_imp=Gimp[:freq_num,3]+Gimp[:freq_num,4]*1j
    # plt.plot(omega,Gloc[:freq_num,1],label='Gloc real')
    # plt.plot(omega,Gloc[:freq_num,2],label='Gloc imag')
    # plt.plot(omega,Gimp[:freq_num,1],label='Gimp real')
    # plt.plot(omega,Gimp[:freq_num,2],label='Gimp imag')
    plt.plot(omega,(1/G_loc-1/G_imp).real,label='1/G_loc-1/G_imp real')
    plt.plot(omega,(1/G_loc-1/G_imp).imag,label='1/G_loc-1/G_imp imag')
    plt.title(' # of iteration: {}'.format(ind))
    plt.legend()
    plt.show()

def compare_Delta(ind):
    if mode ==1:
        filename='./files_pert_ctqmc/{}_{}/pert_Delta.inp.{}'.format(U,T,ind)
    elif mode==0:
        filename='./files_pert_boldc/{}_{}/Delta.inp.{}'.format(U,T,ind+1)
    sigma=np.loadtxt(filename)
    omega=sigma[:,0]
    plt.plot(omega,sigma[:,1],label='11 real pert_Delta.inp')
    plt.plot(omega,sigma[:,2],label='11 imag pert_Delta.inp')
    plt.plot(omega,sigma[:,3],label='22 real pert_Delta.inp')
    plt.plot(omega,sigma[:,4],label='22 imag pert_Delta.inp')
    if mode ==1:
        filename='./files_ctqmc/{}_{}/ori_Delta.inp.{}'.format(U,T,ind)
    if mode ==0:
        filename='./files_boldc/{}_{}/ori_Delta.inp.{}'.format(U,T,ind+1)
    sigma=np.loadtxt(filename)
    omega=sigma[:,0]
    plt.plot(omega,sigma[:,1],label='11 real Delta.inp')
    plt.plot(omega,sigma[:,2],label='11 imag Delta.inp')
    plt.plot(omega,sigma[:,3],label='22 real Delta.inp')
    plt.plot(omega,sigma[:,4],label='22 imag Delta.inp')
    plt.legend()
    plt.show()

def burst_scatter_sig(num,checkpoint,interval=1):
    ilist=np.arange(num)
    # checkpoint=0
    for i in ilist:
        if i%interval==0:
            if mode==0:
                filename='../files_boldc/0_{}_{}/Sig.OCA.{}'.format(U,T,int(i+1))
                # filename='../files_boldc/0_{}_{}/Sig.out.{}'.format(U,T,int(i+1))
            elif mode==1:
                filename='../files_ctqmc/0_{}_{}/ori_Sig.out.{}'.format(U,T,int(i))
        
            if os.path.isfile(filename):
                sigma=np.loadtxt(filename)
                # omega=sigma[:,0]
                plt.scatter(i,sigma[checkpoint,1],c='red')
                plt.scatter(i,sigma[checkpoint,3],c='red')
            else:
                print('cannot find {}'.format(filename))


            # if mode ==0:
            #     filename='../files_boldc/2_{}_{}/Sig.OCA.{}'.format(U,T,int(i+1))
            #     # filename='../files_boldc/2_{}_{}/Sig.out.{}'.format(U,T,int(i+1))
            # elif mode ==1:
            #     filename='../files_ctqmc/2_{}_{}/Sig.out.{}'.format(U,T,int(i))
            # if os.path.isfile(filename):
            #     sigma=np.loadtxt(filename)
            #     # omega=sigma[:,0]
            #     plt.scatter(i,sigma[checkpoint,1],c='blue')
            #     plt.scatter(i,sigma[checkpoint,3],c='blue')
            # else:
            #     print('cannot find {}'.format(filename))

            # if mode ==0:
            #     filename='../files_boldc/3_{}_{}/Sig.OCA.{}'.format(U,T,int(i+1))
            #     # filename='../files_boldc/3_{}_{}/Sig.out.{}'.format(U,T,int(i+1))
            # elif mode ==1:
            #     filename='../files_ctqmc/3_{}_{}/Sig.out.{}'.format(U,T,int(i))
            # if os.path.isfile(filename):
            #     sigma=np.loadtxt(filename)
            #     # omega=sigma[:,0]
            #     plt.scatter(i,sigma[checkpoint,1],c='green')
            #     plt.scatter(i,sigma[checkpoint,3],c='green')
            # else:
            #     print('cannot find {}'.format(filename))

    plt.title('Sigma_imp(om->0).real U={},T={}'.format(U,T))
    plt.xlabel('DMFT iterations Red:DMFT Blue:DMFT+pert2 Green:DMFT+pert3')
    plt.show()
    return 0

def burst_variational(num,checkpoint,interval=1):
    ilist=np.arange(num)
    # checkpoint=0
    for i in ilist:
        if i%interval==0:
            # filename='../files_variational/{}_{}_{}/Sig.OCA.{}'.format(B,U,T,int(i+1))
            filename='../files_variational/{}_{}_{}/Sig.out.{}'.format(B,U,T,int(i+1))
            if os.path.isfile(filename):
                sigma=np.loadtxt(filename)
                # omega=sigma[:,0]
                plt.scatter(i,sigma[checkpoint,1],c='red')
                plt.scatter(i,sigma[checkpoint,3],c='blue')
            else:
                print('cannot find {}'.format(filename))
    plt.title('Sigma_imp(om->0).real B={},U={},T={}'.format(B,U,T))
    # plt.xlabel('DMFT iterations Red:DMFT Blue:DMFT+pert2 Green:DMFT+pert3')
    plt.show()
    return 0

def mag_vs_B(U,T,b_arr):
    # bsize=20
    # b_arr=(np.arange(bsize))/500
    m_arr=np.zeros((4,b_arr.size))
    mb0_arr=np.zeros((4,b_arr.size))
    for i in np.arange(b_arr.size):
        B=b_arr[i]
        # #DMFT data
        # filename0='../files_variational/{}_{}_{}/Sig.OCA.{}'.format(B,U,T,50)
        # if os.path.isfile(filename0):
        #     sigma=np.loadtxt(filename0).T
        #     om=sigma[0,:]
        #     m_arr[0,i]=(sigma[1,-1]-sigma[3,-1])/U
        #     # plt.scatter(B,(sigma[-1,1]-sigma[-1,3])/U,c='red')
        # else:
        #     print('cannot find {}'.format(filename0))
        #perturbation data. order=0 accounts for just DMFT
        for order in (np.arange(4)):# from 0th(DMFT) to n-1th
            
            filename='../dressed_hartree/data/{}_{}_{}_{}_countB0.dat'.format(B,U,T,order)#
            
            if os.path.isfile(filename):
                data=np.loadtxt(filename).T
                m_arr[order,i]=data[0,-1]
            else:
                print('cannot find {}'.format(filename))
            # if order>1:
            #     filename='../dressed_hartree/dataB0/{}_{}_{}_{}.dat'.format(B,U,T,order)
            #     if os.path.isfile(filename):
            #         data=np.loadtxt(filename).T
            #         mb0_arr[order,i]=data[0,-1]
            #     else:
            #         print('cannot find {}'.format(filename))
    plt.plot(b_arr,m_arr[0],label='DMFT')
    plt.plot(b_arr,m_arr[1],label='1st order')
    plt.plot(b_arr,m_arr[2],label='2nd order')
    # plt.plot(b_arr,mb0_arr[2],label='2nd b0 order')
    plt.plot(b_arr,m_arr[3],label='3rd order')
    # plt.plot(b_arr,mb0_arr[3],label='3rd b0 order')
    plt.legend()
    plt.xlabel('B field')
    plt.ylabel('Magnetization')
    plt.title('magntization vs B: U={},T={}'.format(U,T))
    plt.show()


def susceptibility(U):
    b_arr=(np.arange(2))/500
    m_arr=np.zeros((4,b_arr.size))
    Tlist1=np.array([0.2,0.25])
    Tlist=np.concatenate((Tlist1, (np.arange(30)+30)/100), axis=0)
    xi_arr=np.zeros((4,Tlist.size))
    for iT,T in enumerate(Tlist):
        m_arr=np.zeros((4,b_arr.size))
        for i in np.arange(b_arr.size):
            B=b_arr[i]
            for order in (np.arange(4)):# from 0th(DMFT) to n-1th
                filename='../dressed_hartree/data/{}_{}_{}_{}_countB0.dat'.format(B,U,T,order)#
                if os.path.isfile(filename):
                    data=np.loadtxt(filename).T
                    m_arr[order,i]=data[0,-1]
                else:
                    print('cannot find {}'.format(filename))
        for order in (np.arange(4)):
            xi_arr[order,iT]=-(-m_arr[order,0]+m_arr[order,1])/500 # estimation of 1rd order
            # xi_arr[order,iT]=-(-11*m_arr[order,0]+18*m_arr[order,1]-9*m_arr[order,2]+2*m_arr[order,3])/6/500 # estimation up to 3rd order
    plt.plot(Tlist,xi_arr[0],label='DMFT')
    plt.plot(Tlist,xi_arr[1],label='1st')
    plt.plot(Tlist,xi_arr[2],label='2nd')
    plt.plot(Tlist,xi_arr[3],label='3rd')
    plt.legend()
    plt.xlabel('T')
    plt.ylabel('Xi')
    plt.title('Staggered Suseptibility vs T: U={}'.format(U))
    plt.show()
    return 0


if __name__ == "__main__":
    mode=0
    #boldc=0, ctqmc=1
    U=7.0
    T=0.4
    B=0.0
    b_arr=(np.arange(10))/500
    Blist_35=(np.arange(8)+27)/1000
    Blist_40=(np.arange(15)+10)/2000
    b_arr25=(np.arange(20))/500
    b_arr30=(np.arange(40))/500
    b_arr35=np.concatenate(((np.arange(13))/500, Blist_35), axis=0)
    b_arr40=np.concatenate(((np.arange(2))/500, Blist_40), axis=0)
    # burst_variational(50,20,1)# Here I chose Sigma(inf) because I want to check if the particle number is reasonable.
    # mag_vs_B(U,T,b_arr)
    susceptibility(U)
    # single_mode(10)
    # for i in np.arange(50):
    #     compare_G(i+1)