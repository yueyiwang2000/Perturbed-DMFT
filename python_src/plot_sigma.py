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


def find_last_value(file_path,name):
    """
    Search for the last occurrence of 'Fimp=' in a file and return the float value following it.
    
    :param file_path: Path to the file to be searched.
    :return: The float value following the last occurrence of 'Fimp=' or None if not found.
    """
    last_fimp_value = None
    second_last_fimp_value = None
    if os.path.exists(file_path)==False:
        print('cannnot find the file {} !'.format(file_path))
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if name in line:
                    parts = line.split()
                    for part in parts:
                        if part.startswith(name):
                            # Try to extract and convert the number following 'Fimp='
                            try:
                                second_last_fimp_value = last_fimp_value
                                last_fimp_value = float(part.split('=')[1])
                            except ValueError:
                                # If conversion fails, continue to the next occurrence
                                continue
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    # if last_fimp_value==None:
    #     print('cannot find {} in {}'.format(name,file_path))
    return last_fimp_value


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

def susceptibility(U):
    b_arr=(np.arange(2))/500
    m_arr=np.zeros((4,b_arr.size))
    # Tlist1=np.array([0.2,0.25])
    # Tlist=np.concatenate((Tlist1, (np.arange(25)+30)/100), axis=0)
    Tlist=(np.arange(25)+30)/100
    # Tlist=(np.arange(40)+80)/200
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

def plot_sigma(filename,index):
    sigma=np.loadtxt(filename)
    omega=sigma[:,0]
    # for i in np.arange(np.size(sigma[1,:])-1)+1:
    #     plt.plot(omega,sigma[:,i],label='{}th column in {}th file'.format(i,index))
    plt.plot(omega,sigma[:,1],label='real1 in {}th file'.format(index))
    plt.plot(omega,sigma[:,3],label='real2 in {}th file'.format(index))
    # plt.plot(omega,np.zeros_like(omega),label='ZERO')
    plt.legend()
    # plt.xlim((0,omega[-1]/10))
    # plt.ylim((-0.2,0.2))

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
            # filename='../files_variational/{}_{}_{}/Sig.out.{}'.format(B,U,T,int(i+1))
            # filename='../files_mixing/{}_{}_{}/Sig.out.{}'.format(U,T,B,int(i+1))
            # filename='../files_ctqmc/{}_{}/Sig.out.{}'.format(U,T,int(i))
            filename='../files_boldc/{}_{}/Sig.out.{}'.format(U,T,int(i+1))
            if os.path.isfile(filename):
                sigma=np.loadtxt(filename)
                # omega=sigma[:,0]
                plt.scatter(i,sigma[checkpoint,1],c='red')
                plt.scatter(i,sigma[checkpoint,3],c='blue')
                print('find file {}'.format(filename))
            else:
                print('cannot find {}'.format(filename))
    plt.title('Sigma_imp(om->{}).real U={},T={}'.format(checkpoint,U,T))
    plt.xlabel('DMFT iteration #')
    plt.show()
    return 0

def display_file(filename,index):
    sigma=np.loadtxt(filename)
    omega=sigma[:,0]
    plt.plot(omega,sigma[:,1],label='real1 in {}th file'.format(index))
    plt.plot(omega,sigma[:,2],label='imag1 in {}th file'.format(index))
    plt.plot(omega,sigma[:,3],label='real2 in {}th file'.format(index))
    plt.plot(omega,sigma[:,4],label='imag2 in {}th file'.format(index))
    plt.legend()
    plt.title(filename)
    plt.show()

def check_files(U,T,num):
    mode='ctqmc'
    # filechecked='Sig.out'
    # filechecked='delta.inp'
    filechecked='Gf.out'
    for i in np.arange(num):
        filename='../files_{}/{}_{}/{}.{}'.format(mode,U,T,filechecked,int(i))
        display_file(filename,i)
    return 0

    



def checklocimp(U,T,Barr):
    '''
    check how different are Gloc and Gimp at different B.
    '''
    diff1=np.zeros(Barr.size)
    diff2=np.zeros(Barr.size)
    for i,B in enumerate(Barr):
        filename='../files_mixing/{}_{}_{}/diff.dat'.format(U,T,B)
        if os.path.isfile(filename):
            data=np.loadtxt(filename)
            diff1[i]=data[-1,1]
            diff2[i]=data[-1,2]
        else:
            print('cannot find {}'.format(filename))
    plt.plot(Barr,diff1,label='Gimp-Gloc')
    plt.plot(Barr,diff2,label='Gimp-prevGimp=')#optional
    plt.legend()
    plt.xlabel('B')
    plt.ylabel('difference')
    plt.title('difference between Gloc and Gimp: U={},T={}'.format(U,T))
    plt.show()
    return 0

def checklogG(U,T,b_arr,myorder):
    logG_arr=np.zeros((2,4,b_arr.size))
    for i in np.arange(b_arr.size):
        B=b_arr[i]
        for order in (np.arange(4)):# from 0th(DMFT) to n-1th
            
            filename2='../dressed_hartree/data_mixing/{}_{}_{}_{}.dat'.format(U,T,B,order)
            if os.path.isfile(filename2):
                filename=filename2 # usually mixing is better. try that first.
            else:
                print('cannot find {}'.format(filename))    
            
            logG_arr[1,order,i]=find_last_value(filename,'TrlogG_alter=')
            logG_arr[0,order,i]=find_last_value(filename,'TrlogG=')
    # plt.plot(b_arr,logG_arr[0,0],label='DMFT')
    # plt.plot(b_arr,logG_arr[0,1],label='1st order')
    # plt.plot(b_arr,logG_arr[0,2],label='2nd order')
    # plt.plot(b_arr,logG_arr[0,3],label='3rd order')
    # plt.plot(b_arr,logG_arr[1,0],label='DMFT_alter')
    # plt.plot(b_arr,logG_arr[1,1],label='1st order_alter')
    # plt.plot(b_arr,logG_arr[1,2],label='2nd order_alter')
    # plt.plot(b_arr,logG_arr[1,3],label='3rd order_alter')
    plt.plot(b_arr,logG_arr[0,myorder],label='order {}'.format(myorder))
    plt.plot(b_arr,logG_arr[1,myorder],label='order alter {}'.format(myorder))
    plt.legend()
    plt.xlabel('B field')
    plt.ylabel('TrlogG')
    plt.title('TrlogG vs B: U={},T={}'.format(U,T))
    plt.show()
    return 0


if __name__ == "__main__":
    mode=1
    #boldc=0, ctqmc=1


    U=12.0
    T=0.3


    # checklocimp(U,T,b_arr)
    burst_variational(100,-1,1)
    # check_files(U,T,10)
    # checklogG(U,T,b_arr,2)
