import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess
from pert_DMFT_PM import run_perturbation
sys.path.append('../python_src/')
from perturb_lib import *
import pert_DMFT_PM
'''
this Python script aims to calcualate a few simple observables, including Neel Temperature and magnetization, free energies, etc.
'''


def Neel_Temp(order=3,ifplot=1):
    '''
    use the criteria to find the Neel temperature.
    '''
    # T_bound=np.array(((3.0,0.07,0.14),(4.,0.15,0.25),(5.,0.2,0.31),(6.,0.27,0.37),(7.,0.25,0.4),(8.,0.3,0.45),(9.,0.27,0.5),
    #                   (10.,0.3,0.5),(11.,0.3,0.5),(12.,0.3,0.5),(13.,0.3,0.5),(14.,0.25,0.45)))
    T_bound=np.array(((3.0,0.07,0.14),(5.,0.2,0.31),(8.,0.25,0.45),
                      (10.,0.25,0.5),(12.,0.25,0.5),(14.,0.25,0.45)))
    # U_arr=np.arange(3,15)
    U_arr=np.array([3.,5.,8.,10.,12.,14.])
    TN_lower=np.zeros_like(U_arr,dtype=float)
    TN_upper=np.ones_like(U_arr,dtype=float)
    for list in T_bound:
        U=list[0]
        # print(np.where(U_arr==U)[0][0])
        ifupperfound=0
        for T in np.arange(int(list[1]*100),int(list[2]*100))/100:
            dir='./magdata/{}_{}.dat'.format(U,T)
            if os.path.exists(dir):
                # print('already have this directory: ', dir)
                data=np.loadtxt(dir)
                if data[0,2]<data[0,3] and data[0,3]<data[0,4]:# order up, mag up ==>AFM
                    if (order==3) or ((order==4) and (data[0,4]<data[0,5])):
                        TN_lower[np.where(U_arr==U)[0][0]]=T+0.005
                        print('U={}: T={} is AFM!'.format(U,T) )
                if data[0,2]>data[0,3] and data[0,3]>data[0,4] and ifupperfound==0:# order up, mag dn ==>PM
                    if (order==3) or ((order==4) and (data[0,4]>data[0,5])):
                        ifupperfound=1
                        TN_upper[np.where(U_arr==U)[0][0]]=T-0.005
                        print('U={}: T={} is PM!'.format(U,T) )
            # else:
                # print('cannot find ',dir)
    TN_esti=(TN_lower+TN_upper)/2
    TN_err=(-TN_lower+TN_upper)/2
    if ifplot:
        # plt.plot(U_arr,TN_esti, marker='o', linestyle='-',label='lower bound')
        plt.errorbar(U_arr, TN_esti, yerr=TN_err, fmt='-o', capsize=5)
        # plt.title('Neel Temperature')
        plt.xlim(0, 15)
        plt.ylim(0, 0.5)
        plt.show()
    return TN_esti,TN_err

def Neel_Temp_diff_orders():
    U_arr=np.array([3.,5.,8.,10.,12.,14.])
    TN3,TNerr3=Neel_Temp(3,0)
    TN4,TNerr4=Neel_Temp(4,0)
    plt.errorbar(U_arr, TN3, yerr=TNerr3, fmt='-o', capsize=5,label='3rd')
    plt.errorbar(U_arr, TN4, yerr=TNerr4, fmt='-o', capsize=5,label='4th')
    plt.xlim(0, 15)
    plt.ylim(0, 0.5)
    plt.legend()
    plt.xlabel('U')
    plt.ylabel('Temperature')
    plt.show()    

    return 0

def mag_vs_order(U,T):
    '''
    read the data from magdata folder and plot this. 
    X axis is perturbation order and Y is magnetization. different curves belongs to different alphas.
    '''
    dir='./magdata/{}_{}.dat'.format(U,T)
    data=np.loadtxt(dir)
    numalpha=np.shape(data)[0]
    numorder=np.shape(data)[1]-1
    order_arr=np.arange(numorder)
    alp_arr=data[1:,0]
    for ialp, alp in enumerate(alp_arr):
        plt.plot(order_arr, data[ialp+1,1:], marker='o', linestyle='-',label='alpha={}'.format(alp))
    plt.title('Magnetization vs Order: U={} T={}'.format(U, T))
    plt.xticks(order_arr)
    # plt.legend()
    plt.show()
    return 0

def mag_with_err(U,T):
    '''
    give best magnetization estimation using bisection.
    '''
    nfreq=500
    ifit=0
    mag_up=0
    mag_lo=0
    # search for upper bound of alpha
    alpup_up=1.5
    alpup_lo=0.01
    alpup_found=0
    while alpup_found==0:
        alpup_mid=0.5*(alpup_up+alpup_lo)
        
        mag1=run_perturbation(U,T,nfreq,1,alpup_mid,ifit)
        mag2=run_perturbation(U,T,nfreq,2,alpup_mid,ifit)
        mag3=run_perturbation(U,T,nfreq,3,alpup_mid,ifit)
        print('alpup_mid=',alpup_mid,'mag=',mag3)
        if mag1>mag2 and mag2>mag3:# still higher than magup
            alpup_up=alpup_mid
        else:
            alpup_lo=alpup_mid
        if alpup_up-alpup_lo<0.001:
            alpup_found=1
            mag_up=mag3

    # search for lower bound of alpha
    alplo_up=1.5
    alplo_lo=0.01
    alplo_found=0
    while alplo_found==0:
        alplo_mid=0.5*(alplo_up+alplo_lo)
        mag1=run_perturbation(U,T,nfreq,1,alplo_mid,ifit)
        mag2=run_perturbation(U,T,nfreq,2,alplo_mid,ifit)
        mag3=run_perturbation(U,T,nfreq,3,alplo_mid,ifit)
        print('alplo_mid=',alplo_mid,'mag=',mag3)
        if mag1<mag2 and mag2<mag3:
            alplo_lo=alplo_mid
        else:
            alplo_up=alplo_mid
        if alplo_up-alplo_lo<0.001:
            alplo_found=1
            mag_lo=mag3
    return mag_up,mag_lo

def crit_mag(U,Tlist):
    '''
    Critical behavior of magnetization at specific U. to be fixed.
    later check it with Heisenberg critical behavior.
    '''
    mag_upper=np.zeros_like(Tlist,dtype=float)
    mag_lower=np.zeros_like(Tlist,dtype=float)
    for iT, T in enumerate(Tlist):
        print('T=',T)
        mag_upper[iT],mag_lower[iT]=mag_with_err(U,T)
    mag_ave=0.5*(mag_upper+mag_lower)
    mag_err=0.5*(mag_upper-mag_lower)
    # plt.errorbar(Tlist, mag_ave, yerr=mag_err, fmt='-o', capsize=5)
    # plt.xlabel('T')
    # plt.ylabel('magnetization')
    # plt.show()

    filename='./magdata/{}.dat'.format(U)
    f = open(filename, 'w')
    for iT, T in enumerate(Tlist):
        print('{:.2f} {:.5f} {:.5f}'.format(T,mag_ave[iT],mag_err[iT]), file=f)
    f.close()   
    return 0

def DMFT_mag(U,Tlist):
    '''
    magnetization at specific U and different temepratures. This is used to check the critical behavior of DMFT.
    '''
    magarr=np.zeros_like(Tlist,dtype=float)
    nfreq=500
    mu=U/2
    knum=10
    
    for iT, T in enumerate(Tlist):
        beta=1/T
        dir1='../files_DMFT/{}_{}/Sig.out'.format(U,T)
        dir2='../files_DMFT/{}_{}/Sig.OCA'.format(U,T)
        filename1=pert_DMFT_PM.readDMFT(dir1)
        filename2=pert_DMFT_PM.readDMFT(dir2)
        if (os.path.exists(filename1)):
            filename=filename1
        elif (os.path.exists(filename2)):
            filename=filename2
        else:
            print('cannot find both file!',filename1,filename2)
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
        iom=np.concatenate((om[::-1],om))*1j
        z_1=z4D(beta,mu,Sigma11,knum,nfreq)#z-delta
        z_2=z4D(beta,mu,Sigma22,knum,nfreq)#z+delta
        G11_iom,G12_iom=G_iterative(knum,z_1,z_2,np.zeros_like(Sigma11,dtype=complex))
        G22_iom=-G11_iom.conjugate()
        n1=particlenumber4D(G11_iom,beta)
        n2=particlenumber4D(G22_iom,beta)
        magarr[iT]=np.abs(n1-n2)
    return magarr

def plot_mag_critical():
    '''
    plot the  critical behavior of magnetization.
    '''
    dir4='./magdata/4.0.dat'
    data4=np.loadtxt(dir4)
    Tlist4=data4[:,0]
    magave4=data4[:,1]
    magerr4=data4[:,2]
    magDMFT4=DMFT_mag(4.0,Tlist4)

    dir8='./magdata/8.0.dat'
    data8=np.loadtxt(dir8)
    Tlist8=data8[:,0]
    magave8=data8[:,1]
    magerr8=data8[:,2]
    TlistDMFT8=np.arange(20,40)/100
    magDMFT8=DMFT_mag(8.0,TlistDMFT8)

    dir10='./magdata/10.0.dat'
    data10=np.loadtxt(dir10)
    Tlist10=data10[:,0]
    magave10=data10[:,1]
    magerr10=data10[:,2]
    TlistDMFT10=np.arange(25,50)/100
    magDMFT10=DMFT_mag(10.0,TlistDMFT10)

    dir12='./magdata/12.0.dat'
    data12=np.loadtxt(dir12)
    Tlist12=data12[:,0]
    magave12=data12[:,1]
    magerr12=data12[:,2]
    TlistDMFT12=np.arange(25,50)/100    
    magDMFT12=DMFT_mag(12.0,TlistDMFT12)



    plt.errorbar(Tlist4, magave4, yerr=magerr4, fmt='-o', capsize=5,label='U=4',color='r')
    plt.plot(Tlist4,magDMFT4,color='r', linestyle='--')
    plt.errorbar(Tlist8, magave8, yerr=magerr8, fmt='-o', capsize=5,label='U=8',color='black')
    plt.plot(TlistDMFT8,magDMFT8,color='black', linestyle='--')
    plt.errorbar(Tlist10, magave10, yerr=magerr10, fmt='-o', capsize=5,label='U=10',color='g')
    plt.plot(TlistDMFT10,magDMFT10,color='g', linestyle='--')
    plt.errorbar(Tlist12, magave12, yerr=magerr12, fmt='-o', capsize=5,label='U=12',color='b')
    plt.plot(TlistDMFT12,magDMFT12,color='b', linestyle='--')
    plt.title('Magnetization vs T')
    plt.legend()
    plt.grid()
    plt.xlabel('T')
    plt.ylabel('magnetization')
    plt.show()
    return 0


def vs_alpha(U,T,mode='mag',opt=0):
    '''
    plot the quantity at specific U and T but at different alpha and order.
    '''
    if mode=='mag':
        if opt==1:
            dir='./magdata/{}_{}_AFMSIG.dat'.format(U,T)
        else:
            dir='./magdata/{}_{}.dat'.format(U,T)
        dataarr=np.loadtxt(dir)[:,1:]
    elif mode=='E' or mode=='F' or mode=='S':
        if opt==1:
            dir='./energydata/{}_{}_AFMSIG.dat'.format(U,T)
        else:
            dir='./energydata/{}_{}.dat'.format(U,T)
    if mode=='E':
        dataarr=np.loadtxt(dir)[:,5:]
    elif mode=='F':
        dataarr=np.loadtxt(dir)[:,1:5]
    elif mode=='S':
        dataarr=(np.loadtxt(dir)[:,5:]-np.loadtxt(dir)[:,1:5])/T
    alpha_arr=np.loadtxt(dir)[:,0]

    plt.plot(alpha_arr, dataarr[:,0], marker='o', linestyle='-',label='0th')
    plt.plot(alpha_arr, dataarr[:,1], marker='^', linestyle='-',label='1st')
    plt.plot(alpha_arr, dataarr[:,2], marker='s', linestyle='-',label='2nd')
    plt.plot(alpha_arr, dataarr[:,3], marker='p', linestyle='-',label='3rd')
    plt.plot(alpha_arr, dataarr[:,4], marker='h', linestyle='-',label='4th')
    plt.legend()
    plt.title('{} vs Order: U={} T={}'.format(mode,U, T))
    plt.xlabel('Variational parameter alpha')
    plt.ylabel('magnetization')
    plt.show()

    # plt.plot(alpha_arr, magarr[1]-magarr[0], marker='^', linestyle='-',label='DMFT {} 1st-0th'.format(typelist[ifit]))
    # plt.plot(alpha_arr, magarr[2]-magarr[0], marker='s', linestyle='-',label='DMFT {} 2nd-0th'.format(typelist[ifit]))
    # plt.plot(alpha_arr, magarr[3]-magarr[0], marker='p', linestyle='-',label='DMFT {} 3rd-0th'.format(typelist[ifit]))
    # plt.title('Magnetization vs Order: U={} T={}'.format(U, T))
    # plt.legend()
    # plt.show()
    return 0


def phase_test(mode,ifsigafm=0):
    '''
    This is a brief test of entropy got from different alphas.
    '''
    # U=8.
    # Tlist=np.array([0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58])  #
    U=10.
    Tlist=np.array([0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,
                    0.49,0.5,0.51,0.52,0.53,0.54,0.55])#,0.56,0.57,0.58,0.59,0.6,0.61,0.62
    # U=12.
    # Tlist=np.array([0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,
    #                 0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59])#,
    if ifsigafm==0:  
        dir='./energydata/{}_{}.dat'.format(U,Tlist[0])
    else:
        dir='./energydata/{}_{}_AFMSIG.dat'.format(U,Tlist[0])
    alphastart=0
    alpha_arr=np.loadtxt(dir)[alphastart:,0]
    Slist=np.zeros((alpha_arr.size,Tlist.size))
    for iT,T in enumerate(Tlist):
        dir='./energydata/{}_{}.dat'.format(U,T)
        if mode=='S':
            Slist[:,iT]=(np.loadtxt(dir)[alphastart:,8]-np.loadtxt(dir)[alphastart:,4])/T# S=(E-F)/T
        elif mode=='E':
            Slist[:,iT]=np.loadtxt(dir)[alphastart:,8]# total energy
        elif mode=='F':
            Slist[:,iT]=np.loadtxt(dir)[alphastart:,4]# free energy
    for ialp, alpha in enumerate(alpha_arr):
        plt.plot(Tlist,Slist[ialp],label='alpha={}'.format(alpha))
    plt.legend()
    plt.title('{} vs. T at different alphas. U={}, 3rd order'.format(mode,U))
    plt.xlabel('T')
    plt.ylabel(mode)
    plt.show()
    for ialp, alpha in enumerate(alpha_arr):
        plt.plot((Tlist[1:]+Tlist[:-1])/2,(Slist[ialp,1:]-Slist[ialp,:-1])/0.01,label='alpha={}'.format(alpha))
    plt.legend()
    plt.title('d{}/dT vs. T at different alphas. U={}, 3rd order'.format(mode,U))
    plt.xlabel('T')
    plt.ylabel('d{}/dT'.format(mode))
    plt.show()    

    for ialp, alpha in enumerate(alpha_arr):
        plt.plot((Tlist[1:]+Tlist[:-1])/2,-(Slist[ialp,1:]-Slist[ialp,:-1])/0.01,label='alpha={}'.format(alpha))
    plt.legend()
    plt.title('-d{}/dT vs. T at different alphas. U={}, 3rd order'.format(mode,U))
    plt.xlabel('T')
    plt.ylabel('-d{}/dT'.format(mode))
    plt.show()        
    return 0


if __name__ == "__main__":
    # phase_test('E')
    # vs_alpha(5.0,0.2,'mag',0)
    # Neel_Temp(3)
    Neel_Temp_diff_orders()

