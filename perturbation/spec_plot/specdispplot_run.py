import  os,sys,subprocess
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../')
import perturb_lib as lib
from scipy.interpolate import interp1d
from matplotlib.colors import LightSource
import matplotlib.colors as colors
import time
from scipy.signal import savgol_filter
import fft_convolution as fft
import copy
import scipy
from scipy.optimize import curve_fit
import specplot_run
from maxentropy import Pade
import matplotlib.cm as cm
'''
This function generalize the workflow of plotting spectrum function. Here are the precedures:
1. choose a few well-converged self energy files (on imag axis) and do the average. 
(1.5. for accuracy, extract the Sigma_inffreq before the first step. then put it back after the second step.)
2. call maxent_run.py to use the max entropy method to do the continuation to get self energy on real axis.
3. Using G=1/(omega+mu-eps_k-Sigma) to get green's function. Remember in this step we are all in the real axis!
4. Spectrum function is A=-1/pi Im(G). Here we have to choose our kpath.
5. Use imshow() to generate a heat map. Sometimes have to tune the brightness.

input example:     python specplot_run.py 0 7.0 0.4 0
'''

fileSig='Sig.out'
fileaveSig='sig.inpx'
filemodSig='sigmod.inpx'
filerealSig='Sig.out'


#Don't get confused. fileSig is stored in another dir so it  won't get confused with fileoutsig in practice.
# sizefont=18# default:12
# plt.rc('font', size=sizefont) 
# plt.rc('axes', titlesize=sizefont) 
# plt.rc('axes', labelsize=sizefont) 
# plt.rc('xtick', labelsize=sizefont)
# plt.rc('ytick', labelsize=sizefont)

# about filename of sigma:
# if calculated from only 2 orders of boldc solver, named Sig.OCA;
# else, named Sig.out

# fileSig='Sig.out'
# fileSig='Sig.OCA'#sometimes for bold impurity solver without mc, the filename is Sig.OCA

def single_gaussian(x, A1, mu1, sigma1):
    g1 = A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
    return g1


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

def Fourier_Basis(n1,n2,n3,a=1,t=1,knum=10):
    k1,k2,k3=lib.gen_full_kgrids(knum)
    kx=(-k1+k2+k3)*np.pi/a
    ky=(k1-k2+k3)*np.pi/a
    kz=(k1+k2-k3)*np.pi/a
    e_k=np.cos(kx*n1)*np.cos(ky*n2)*np.cos(kz*n3)
    fac=np.array([np.sqrt(1/knum),np.sqrt(2/knum)])
    fac1 = fac[1] if n1 > 0 else fac[0]
    fac2 = fac[1] if n2 > 0 else fac[0]
    fac3 = fac[1] if n3 > 0 else fac[0]
    return e_k*fac1*fac2*fac3

def Fourier_Basisat1k(n1,n2,n3,u,v,w,a=1,t=1,knum=10):
    '''
    k=uk1+vk2+wk3.
    '''
    kx=(-u+v+w)*np.pi/a
    ky=(u-v+w)*np.pi/a
    kz=(u+v-w)*np.pi/a
    e_k=np.cos(kx*n1)*np.cos(ky*n2)*np.cos(kz*n3)
    fac=np.array([np.sqrt(1/knum),np.sqrt(2/knum)])
    fac1 = fac[1] if n1 > 0 else fac[0]
    fac2 = fac[1] if n2 > 0 else fac[0]
    fac3 = fac[1] if n3 > 0 else fac[0]
    return e_k*fac1*fac2*fac3

def gen_kbasis_new(Nmax,knum=10):
    '''
    basis based on Fourier basis.  because of the symmetry, we have:
    cos(n1*kx)cos(n2*ky)cos(n3*kz) as basis. ni=0,1,2,....Nmax-1.
    '''
    indlist=np.zeros((Nmax**3,3),dtype=int)
    for nx in np.arange(Nmax):
        for ny in np.arange(Nmax):
            for nz in np.arange(Nmax):
                if nx+ny+nz<=Nmax:
                    indlist[nx*Nmax**2+ny*Nmax+nz,:]=np.array([nx,ny,nz])
    sorted_array = -np.sort(-indlist, axis=1)
    indlistreduced=np.unique(sorted_array,axis=0)
    # print(indlistreduced)
    basisnum=np.shape(indlistreduced)[0]
    # print('number of basis:',basisnum)

    sorted_indices = np.argsort(np.sum(indlistreduced,axis=1))
    indlist_final=indlistreduced[sorted_indices,:]
    # print(indlist_final)

    kbasis=np.zeros((2,basisnum,knum,knum,knum))
    innerp=np.zeros(basisnum)
    for i in np.arange(basisnum):
        nx=indlist_final[i,0]
        ny=indlist_final[i,1]
        nz=indlist_final[i,2]
        kbasis[np.sum(indlist_final[i])%2,i,:,:,:]=(Fourier_Basis(nx,ny,nz)+Fourier_Basis(nx,nz,ny)+Fourier_Basis(ny,nx,nz)+Fourier_Basis(ny,nz,nx)+Fourier_Basis(nz,ny,nx)+Fourier_Basis(nz,nx,ny))/6
        # kbasis[np.sum(indlist_final[i])%2,i,:,:,:]=Fourier_Basis(nx,ny,nz)
        #sometimes have to be normalized
        innerp[i]=np.sum(kbasis[np.sum(indlist_final[i])%2,i,:,:,:]*kbasis[np.sum(indlist_final[i])%2,i,:,:,:])
        # print('innerp of {} {} {}:{}'.format(nx,ny,nz,innerp))
        kbasis[np.sum(indlist_final[i])%2,i,:,:,:]/=np.sqrt(innerp[i])

    return kbasis,innerp

def kbasis_at_a_k(Nmax,u,v,w,innerp,knum=10):
    '''
    basis based on Fourier basis.  because of the symmetry, we have:
    cos(n1*kx)cos(n2*ky)cos(n3*kz) as basis. ni=0,1,2,....Nmax-1.
    here we support any float input of u,v,w. for the k path.
    '''
    indlist=np.zeros((Nmax**3,3),dtype=int)
    for nx in np.arange(Nmax):
        for ny in np.arange(Nmax):
            for nz in np.arange(Nmax):
                if nx+ny+nz<=Nmax:
                    indlist[nx*Nmax**2+ny*Nmax+nz,:]=np.array([nx,ny,nz])
    sorted_array = -np.sort(-indlist, axis=1)
    indlistreduced=np.unique(sorted_array,axis=0)
    # print(indlistreduced)
    basisnum=np.shape(indlistreduced)[0]
    # print('number of basis:',basisnum)

    sorted_indices = np.argsort(np.sum(indlistreduced,axis=1))
    indlist_final=indlistreduced[sorted_indices,:]
    # print(indlist_final)

    kbasis=np.zeros((2,basisnum))
    for i in np.arange(basisnum):
        nx=indlist_final[i,0]
        ny=indlist_final[i,1]
        nz=indlist_final[i,2]
        # allkbasis=(Fourier_Basis(nx,ny,nz)+Fourier_Basis(nx,nz,ny)+Fourier_Basis(ny,nx,nz)+Fourier_Basis(ny,nz,nx)+Fourier_Basis(nz,ny,nx)+Fourier_Basis(nz,nx,ny))/6
        kbasis[np.sum(indlist_final[i])%2,i]=(Fourier_Basisat1k(nx,ny,nz,u,v,w)+Fourier_Basisat1k(nx,nz,ny,u,v,w)+Fourier_Basisat1k(ny,nx,nz,u,v,w)+Fourier_Basisat1k(ny,nz,nx,u,v,w)+Fourier_Basisat1k(nz,ny,nx,u,v,w)+Fourier_Basisat1k(nz,nx,ny,u,v,w))/6
        # kbasis[np.sum(indlist_final[i])%2,i,:,:,:]=Fourier_Basis(nx,ny,nz)
        #sometimes have to be normalized
        # innerp=np.sum(allkbasis[np.sum(indlist_final[i])%2,i,:,:,:]*allkbasis[np.sum(indlist_final[i])%2,i,:,:,:])
        # print('innerp of {} {} {}:{}'.format(nx,ny,nz,innerp))
        kbasis[np.sum(indlist_final[i])%2,i]/=np.sqrt(innerp[i])

    return kbasis

def restore_tk1(cli,u,kbasis1pt,knum=10):#G(t,k)=\Sum_{l,i} u_l(t) m_i(k)
    cl=np.sum(cli[:,:]*kbasis1pt[None,:],axis=(1))#/knum**3
    # print('shapeof clk:',np.shape(clk))
    Gt=np.sum(cl[:,None]*u[:,:],axis=(0))
    return Gt


def dist_kpoints(k1,k2):
    k1x=-k1[0]+k1[1]+k1[2]
    k1y=k1[0]-k1[1]+k1[2]
    k1z=k1[0]+k1[1]-k1[2]
    k2x=-k2[0]+k2[1]+k2[2]
    k2y=k2[0]-k2[1]+k2[2]
    k2z=k2[0]+k2[1]-k2[2]
    dist=np.sqrt((k1x-k2x)**2+(k1y-k2y)**2+(k1z-k2z)**2)
    return dist

def gen_entire_kpath(highsym_path,kpoints_per_dist=5):
    numkpoints=np.shape(highsym_path)[0]
    entirepath=np.array([[highsym_path[0,0],highsym_path[0,1],highsym_path[0,2]]])#entire kpath
    high_sym_index=np.zeros(numkpoints)#indices of high symmetry points. used later.
    # print(entirepath)
    for i in np.arange(numkpoints-1):
        k1=highsym_path[i]
        k2=highsym_path[i+1]
        numbetw=int(dist_kpoints(k1,k2)*kpoints_per_dist)
        k_arrays = np.vstack([np.linspace(k1[j], k2[j], numbetw+1) for j in range(3)]).T
        entirepath=np.vstack((entirepath,k_arrays[1:]))
        high_sym_index[i+1]=np.shape(entirepath)[0]-1
    # print(entirepath)
    # print(high_sym_index)
    return entirepath,high_sym_index

def findyticks(ene,ylist):
    yticks=np.zeros_like(ylist)
    i=0
    for j in np.arange(ene.size):
        if i<ylist.size and ene[j]<ylist[i] :
            yticks[i]=j
            i+=1
    return yticks

def calc_real_axis(U,T,order,u,v,w,ifpade,ttpts,indexofpt,alpha,ifsave=1):
    '''
    K=ub1+vb2+wb3. K is the point in 1BZ, bi are reciprocal vectors.
    ttpts=total number of k points
    indexofpt= the index of this point. 1=first, 2=second,...

    Note: this function is trying to use the Mk to do analytical continuation.
    '''

    # for perturbation
    # alpha=0.01# or 0.05
    knum=10
    beta=1/T 
    if order==0:# HAVE TO BE COMPLETED! 0TH ORDER
        print('please use specplot_run.py!')
    else:

        foldernum=1
        foldernum2='_search'
        dir11='../Sigma_disp{}/{}_{}/{}_{}_{}_{}_11.dat'.format(foldernum,U,T,U,T,order,alpha)
        dir12='../Sigma_disp{}/{}_{}/{}_{}_{}_{}_12.dat'.format(foldernum,U,T,U,T,order,alpha)
        dir11c='../Sigma_disp{}/{}_{}/{}_{}_{}_{}_11const.dat'.format(foldernum,U,T,U,T,order,alpha)
        data11=np.loadtxt(dir11)
        data12=np.loadtxt(dir12)
        data11c=np.loadtxt(dir11c)

        imax=4
        filenameu='../Sigma_imp/taubasis.txt'
        ut=np.loadtxt(filenameu).T 
        taunum=np.shape(ut)[1]-1
        nfreq=500
        taulist=(np.arange(taunum+1))/taunum*beta
        ori_grid=(np.arange(nfreq*2)+0.5)/(nfreq*2)*beta
        allkbasis,innerp=gen_kbasis_new(imax)
        kbasis=kbasis_at_a_k(imax,u,v,w,innerp)
        Sig11=restore_tk1(data11,ut,kbasis[0])
        Sig12=restore_tk1(data12,ut,kbasis[1])
        Sig11const=np.sum(data11c*kbasis[0])
        Sig22const=U-Sig11const
        interpolator_11 = interp1d(taulist, Sig11, kind='cubic', axis=0, fill_value='extrapolate')
        interpolator_12 = interp1d(taulist, Sig12, kind='cubic',  axis=0,fill_value='extrapolate')
        Sig11tk_full=interpolator_11(ori_grid)
        Sig12tk_full=interpolator_12(ori_grid)
        Sigiom_11=fft.fermion_ifft(Sig11tk_full,beta)
        Sigiom_12=fft.fermion_ifft(Sig12tk_full,beta)
        Sigiom_22=-Sigiom_11.conjugate()
    # plt.plot(Sigiom_11.real,label='11 real')
    # plt.plot(Sigiom_11.imag,label='11 imag')
    # plt.plot(Sigiom_12.real,label='12 real')
    # plt.plot(Sigiom_12.imag,label='12 imag')
    # plt.legend()
    # plt.title('u={} v={} w={}'.format(u,v,w))
    # plt.show()

    # after getting sigma_k(iom), try to diagonalize it to get SigA and SigB:
    #SigA/B=0.5*(Sig11+Sig22+-sqrt((sig11-sig22)^2+4Sig12^2))
    # SigA=(Sigiom_11+Sig11const+Sigiom_22+Sig22const+np.sqrt((Sigiom_11+Sig11const-Sigiom_22-Sig22const)**2 +4*(Sigiom_12+epsk)**2))/2
    # SigB=(Sigiom_11+Sig11const+Sigiom_22+Sig22const-np.sqrt((Sigiom_11+Sig11const-Sigiom_22-Sig22const)**2 +4*(Sigiom_12+epsk)**2))/2
    SigA=(Sigiom_11+Sigiom_12)
    SigB=(Sigiom_11-Sigiom_12)

    # SigAconst=SigA[-1]
    # SigBconst=SigB[-1]
    om=(2*np.arange(2*nfreq)+1-2*nfreq)*np.pi/beta
    MA=1/(om*1j-SigA)
    MB=1/(om*1j-SigB)
    # sigA_inf=Sf[1,-1]
    # sigB_inf=Sf[3,-1]

    SigAp=(Sigiom_22+Sigiom_12)
    SigBp=(Sigiom_22-Sigiom_12)
    # SigApconst=SigAp[-1]
    # SigBpconst=SigBp[-1]    
    MAp=1/(om*1j-SigAp)
    MBp=1/(om*1j-SigBp)  



    filemodSig='M.inpx'
    f = open(filemodSig, 'w')
    for i,iom in enumerate(om[nfreq:]):
        print(iom, MA[i+nfreq].real, MA[i+nfreq].imag,MB[i+nfreq].real, MB[i+nfreq].imag,MAp[i+nfreq].real, MAp[i+nfreq].imag,MBp[i+nfreq].real, MBp[i+nfreq].imag, file=f) 
        # print(iom, Sigiom_11[i+nfreq].real, Sigiom_11[i+nfreq].imag,Sigiom_22[i+nfreq].real, Sigiom_22[i+nfreq].imag,Sigiom_12[i+nfreq].real, Sigiom_12[i+nfreq].imag, file=f) 
    f.close()
    #step2: run maxent and put sig_inf back
    if ifpade==0:
        cmd_maxent='python maxent_run.py '+filemodSig+' >out.txt'
        subprocess.call(cmd_maxent,shell=True)
        Sfreal = np.loadtxt(filerealSig).T
        omreal=Sfreal[0,2:]#here we start from the 2nd line because the real axis self energy keeps first 2 lines from the imag axis self-energy...idk why but that is what i got.
        MrealA=Sfreal[1,2:]+Sfreal[2,2:]*1j
        MrealB=Sfreal[3,2:]+Sfreal[4,2:]*1j
        MrealAp=Sfreal[5,2:]+Sfreal[6,2:]*1j
        MrealBp=Sfreal[7,2:]+Sfreal[8,2:]*1j
    else:
        omreal = np.linspace(-15,15,1001)
        # gamma=0.001
        # Norder=100
        MrealA=Pade(om[nfreq:],MA[nfreq:],omreal,gamma,Norder)
        MrealB=Pade(om[nfreq:],MB[nfreq:],omreal,gamma,Norder)
        MrealAp=Pade(om[nfreq:],MAp[nfreq:],omreal,gamma,Norder)
        MrealBp=Pade(om[nfreq:],MBp[nfreq:],omreal,gamma,Norder)
    # plt.scatter(om[nfreq:],MA[nfreq:].real,label='MA.real')
    # plt.scatter(om[nfreq:],MA[nfreq:].imag,label='MA.imag')
    # plt.plot(omreal,MrealA.real,label='MAreal.real')
    # plt.plot(omreal,MrealA.imag,label='MAreal.imag')
    # plt.legend()
    # plt.xlim(-15,15)
    # plt.show()
    # plt.scatter(om[nfreq:],MB[nfreq:].real,label='MB.real')
    # plt.scatter(om[nfreq:],MB[nfreq:].imag,label='MB.imag')
    # plt.plot(omreal,MrealB.real,label='MrealB.real')
    # plt.plot(omreal,MrealB.imag,label='MrealB.imag')
    # plt.legend()
    # plt.xlim(-15,15)
    # plt.show()


    sigrealA=omreal-1/MrealA+Sig11const
    sigrealB=omreal-1/MrealB+Sig11const
    sigrealAp=omreal-1/MrealAp+Sig22const
    sigrealBp=omreal-1/MrealBp+Sig22const
    sigreal11=(sigrealA+sigrealB)/2
    sigreal12=(sigrealA-sigrealB)/2
    sigreal22=(sigrealAp+sigrealBp)/2

    # plt.scatter(om,SigA.real,label='SigA real')
    # plt.scatter(om,SigA.imag,label='SigA imag')
    # plt.plot(omreal,sigrealA.real,label='sigrealA real')
    # plt.plot(omreal,sigrealA.imag,label='sigrealA imag')
    # plt.legend()
    # plt.xlim(-20,20)
    # plt.title('SigA FOR u={} v={} w={}'.format(u,v,w))
    # plt.show()

    # plt.scatter(om,SigB.real,label='SigB real')
    # plt.scatter(om,SigB.imag,label='SigB imag')
    # plt.plot(omreal,sigrealB.real,label='sigrealB real')
    # plt.plot(omreal,sigrealB.imag,label='sigrealB imag')
    # plt.legend()
    # plt.xlim(-20,20)
    # plt.title('SigB FOR u={} v={} w={}'.format(u,v,w))
    # plt.show() 

    # plt.scatter(om,Sigiom_12.real,label='Sigiom_12 real')
    # plt.scatter(om,Sigiom_12.imag,label='Sigiom_12 imag')
    # plt.plot(omreal,sig12real.real,label='Sigiom12real real')
    # plt.plot(omreal,sig12real.imag,label='Sigiom12real imag')
    # plt.legend()
    # plt.xlim(-20,20)
    # plt.title('Sig12_eps FOR u={} v={} w={}'.format(u,v,w))
    # plt.show() 


    #Note: This is 
    if ifsave:
        MEMopt=1
        os.makedirs(outputdir, exist_ok=True)
        f = open(outputdir+'{}_{}_{}_{}_{}_{}_{}.dat'.format(U,T,order,MEMopt,alpha,ttpts,indexofpt), 'w')
        # f = open(outputdir+'{}_{}_{}_{}_{}_{}.dat'.format(U,T,order,alpha,ttpts,indexofpt), 'w')
        for i,iom in enumerate(omreal):
            print(iom, sigreal11[i].real, sigreal11[i].imag, sigreal22[i].real, sigreal22[i].imag,sigreal12[i].real, sigreal12[i].imag,  file=f) 
        f.close()
    # try to plot the real axis self-energy

    return omreal,sigreal11,sigreal22,sigreal12# Note: here we return sig11 and sig12!



def calc_real_axis2(U,T,order,u,v,w,ifpade,ttpts,indexofpt,alpha,ifsave=1):
    '''
    K=ub1+vb2+wb3. K is the point in 1BZ, bi are reciprocal vectors.
    ttpts=total number of k points
    indexofpt= the index of this point. 1=first, 2=second,...

    Note: this function is trying to use the Mk to do analytical continuation.
    '''

    # for perturbation
    # alpha=0.01# or 0.05
    beta=1/T 
    if order==0:# HAVE TO BE COMPLETED! 0TH ORDER
        print('please use specplot_run.py!')
    else:

        foldernum=1
        foldernum2='_search'
        dir11='../Sigma_disp{}/{}_{}/{}_{}_{}_{}_11.dat'.format(foldernum,U,T,U,T,order,alpha)
        dir12='../Sigma_disp{}/{}_{}/{}_{}_{}_{}_12.dat'.format(foldernum,U,T,U,T,order,alpha)
        dir11c='../Sigma_disp{}/{}_{}/{}_{}_{}_{}_11const.dat'.format(foldernum,U,T,U,T,order,alpha)
        data11=np.loadtxt(dir11)
        data12=np.loadtxt(dir12)
        data11c=np.loadtxt(dir11c)

        imax=4
        filenameu='../Sigma_imp/taubasis.txt'
        ut=np.loadtxt(filenameu).T 
        taunum=np.shape(ut)[1]-1
        nfreq=500
        taulist=(np.arange(taunum+1))/taunum*beta
        ori_grid=(np.arange(nfreq*2)+0.5)/(nfreq*2)*beta
        allkbasis,innerp=gen_kbasis_new(imax)
        kbasis=kbasis_at_a_k(imax,u,v,w,innerp)
        Sig11=restore_tk1(data11,ut,kbasis[0])
        Sig12=restore_tk1(data12,ut,kbasis[1])
        Sig11const=np.sum(data11c*kbasis[0])
        Sig22const=U-Sig11const
        interpolator_11 = interp1d(taulist, Sig11, kind='cubic', axis=0, fill_value='extrapolate')
        interpolator_12 = interp1d(taulist, Sig12, kind='cubic',  axis=0,fill_value='extrapolate')
        Sig11tk_full=interpolator_11(ori_grid)
        Sig12tk_full=interpolator_12(ori_grid)
        Sigiom_11=fft.fermion_ifft(Sig11tk_full,beta)
        Sigiom_12=fft.fermion_ifft(Sig12tk_full,beta)
        Sigiom_22=-Sigiom_11.conjugate()
    # plt.plot(Sigiom_11.real,label='11 real')
    # plt.plot(Sigiom_11.imag,label='11 imag')
    # plt.plot(Sigiom_12.real,label='12 real')
    # plt.plot(Sigiom_12.imag,label='12 imag')
    # plt.legend()
    # plt.title('u={} v={} w={}'.format(u,v,w))
    # plt.show()

    # after getting sigma_k(iom), try to diagonalize it to get SigA and SigB:
    
    SigA=Sigiom_11+Sigiom_12
    SigB=Sigiom_11-Sigiom_12
    SigC=Sigiom_22+Sigiom_12
    SigD=Sigiom_22-Sigiom_12
    om=(2*np.arange(2*nfreq)+1-2*nfreq)*np.pi/beta



    if ifpade==0:
        f = open(filemodSig, 'w')
        for i,iom in enumerate(om[nfreq:]):
            print(iom, SigA[i+nfreq].real, SigA[i+nfreq].imag,SigB[i+nfreq].real, SigB[i+nfreq].imag,SigC[i+nfreq].real, SigC[i+nfreq].imag,SigD[i+nfreq].real, SigD[i+nfreq].imag, file=f) 
            # print(iom, Sigiom_11[i+nfreq].real, Sigiom_11[i+nfreq].imag,Sigiom_22[i+nfreq].real, Sigiom_22[i+nfreq].imag,Sigiom_12[i+nfreq].real, Sigiom_12[i+nfreq].imag, file=f) 
        f.close()
        #step2: run maxent and put sig_inf back
        cmd_maxent='python maxent_run.py '+filemodSig+' >out.txt'
        subprocess.call(cmd_maxent,shell=True)
        Sfreal = np.loadtxt(filerealSig).T
        omreal=Sfreal[0,2:]#here we start from the 2nd line because the real axis self energy keeps first 2 lines from the imag axis self-energy...idk why but that is what i got.
        sigrealA=Sfreal[1,2:]+Sfreal[2,2:]*1j+Sig11const
        sigrealB=Sfreal[3,2:]+Sfreal[4,2:]*1j+Sig11const
        sigrealC=Sfreal[5,2:]+Sfreal[6,2:]*1j+Sig22const
        sigrealD=Sfreal[7,2:]+Sfreal[8,2:]*1j+Sig22const
    else:#pade of sigma
        omreal = np.linspace(-15,15,1001)

        sigrealA=Pade(om[nfreq:],SigA[nfreq:],omreal,gamma,Norder)+Sig11const
        sigrealB=Pade(om[nfreq:],SigB[nfreq:],omreal,gamma,Norder)+Sig11const
        sigrealC=Pade(om[nfreq:],SigC[nfreq:],omreal,gamma,Norder)+Sig22const
        sigrealD=Pade(om[nfreq:],SigD[nfreq:],omreal,gamma,Norder)+Sig22const



    sig12real=(sigrealA-sigrealB)/2 
    sig11real=(sigrealA+sigrealB)/2 
    sig22real=(sigrealC+sigrealD)/2 



    # plt.scatter(om,SigA.real,label='SigA real')
    # plt.scatter(om,SigA.imag,label='SigA imag')
    # plt.plot(omreal,sigrealA.real,label='sigrealA real')
    # plt.plot(omreal,sigrealA.imag,label='sigrealA imag')
    # plt.legend()
    # plt.xlim(-20,20)
    # plt.title('SigA FOR u={} v={} w={}'.format(u,v,w))
    # plt.show()

    # plt.scatter(om,SigB.real,label='SigB real')
    # plt.scatter(om,SigB.imag,label='SigB imag')
    # plt.plot(omreal,sigrealB.real,label='sigrealB real')
    # plt.plot(omreal,sigrealB.imag,label='sigrealB imag')
    # plt.legend()
    # plt.xlim(-20,20)
    # plt.title('SigB FOR u={} v={} w={}'.format(u,v,w))
    # plt.show() 

    # plt.scatter(om,Sigiom_12.real,label='Sigiom_12 real')
    # plt.scatter(om,Sigiom_12.imag,label='Sigiom_12 imag')
    # plt.plot(omreal,sig12real.real,label='Sigiom12real real')
    # plt.plot(omreal,sig12real.imag,label='Sigiom12real imag')
    # plt.legend()
    # plt.xlim(-20,20)
    # plt.title('Sig12_eps FOR u={} v={} w={}'.format(u,v,w))
    # plt.show() 


    #Note: This is 
    if ifsave:
        if ifpade==0:
            MEMopt=2
        else:
            MEMopt=-2
        os.makedirs(outputdir, exist_ok=True)
        f = open(outputdir+'{}_{}_{}_{}_{}_{}_{}.dat'.format(U,T,order,MEMopt,alpha,ttpts,indexofpt), 'w')
        # f = open(outputdir+'{}_{}_{}_{}_{}_{}.dat'.format(U,T,order,alpha,ttpts,indexofpt), 'w')
        for i,iom in enumerate(omreal):
            print(iom, sig11real[i].real, sig11real[i].imag, sig22real[i].real, sig22real[i].imag,sig12real[i].real, sig12real[i].imag,  file=f) 
        f.close()
    # try to plot the real axis self-energy

    return omreal,sig11real,sig22real,sig12real# Note: here we return sig11 and sig12!

def gen_spec_plot(U,T,order,alpha,MEMopt=1,brightness_factor=1,ifplot=0):
    #ГXWKГL UWLK. THIS IS THE MOST REASONABLE PATH. 
    # k_highsym_path=np.array([[0,0,0],#Г
    #                         [0,1/2,1/2],#X
    #                         [1/4,3/4,1/2],#W
    #                         [3/8,3/4,3/8],#K
    #                         [0,0,0],#Г
    #                         [1/2,1/2,1/2]#L
    #                         ])#http://lampx.tugraz.at/~hadley/ss1/bzones/fcc.php
    #[1/4,5/8,5/8],#U
    debug=0
    k_highsym_path=np.array([[0,0,0],#Г
                            [0,1/2,1/2],#X:kx=pi/a, ky=kz=0
                            [1/2,1/2,1],#M kx=ky=pi/a, kz=0
                            [0,0,0],#Г
                            [1,1,1],#R kx=ky=kz=pi/a
                            [0,1/2,1/2]#X
                            ])      
    klabels=['Г', 'X', 'M', 'Г', 'R',  'X']

    kptperdis=20
    if order==1 or MEMopt<0:
        kptperdis=100
    kpath,highsymindex=gen_entire_kpath(k_highsym_path,kptperdis)
    kptnum=np.shape(kpath)[0]

    print('total kpts in kpath:',kptnum)
    beta=1/T 
    mu=U/2
    # klabels=['Г', 'X', 'W', 'K', 'Г', 'L']
    Emin=-(6+U/2)
    Emax=(6+U/2)
    energynum=1000
    energy_new=np.linspace(Emax, Emin, num=energynum)
    A_A=np.zeros((energynum,kptnum))#energy,k
    A_B=np.zeros((energynum,kptnum))
    A_AB=np.zeros((energynum,kptnum))
    A_off=np.zeros((energynum,kptnum))
    Attl=np.zeros((energynum,kptnum))
    # k1p=kpath[:,0].T
    # k2p=kpath[:,1].T
    # k3p=kpath[:,2].T
    # disp=lib.dispersion(k1p,k2p,k3p)
    # #plot the dispersion
    # plt.plot(disp,color='blue')
    # plt.plot(-disp,color='blue')
    # ax = plt.gca()
    # ax.set_xticks(highsymindex)
    # ax.set_xticklabels(klabels)
    # ax.set_ylabel('Energy/t')
    # for tick in highsymindex:
    #     ax.axvline(x=tick, color='black', linestyle=':')
    # plt.show()


    for ki in np.arange(kptnum):
        if order>1 and MEMopt>0:
            print('ki={}/{}'.format(ki,kptnum))
        k1=kpath[ki,0]# coefficients of linear combination of reciprocal vectors
        k2=kpath[ki,1]
        k3=kpath[ki,2]
        disp=lib.dispersion(k1,k2,k3)
        dirSreal=outputdir+'{}_{}_{}_{}_{}_{}_{}.dat'.format(U,T,order,MEMopt,alpha,kptnum,ki)
        # dirSreal=outputdir+'{}_{}_{}_{}_{}_{}_{}.dat'.format(U,T,order,MEMopt,alpha,ttpts,indexofpt)
        # brightness_factor=1
        # om,sigrealA,sigrealB,sigoffdiag=calc_real_axis(U,T,order,k1,k2,k3,disp,kptnum,ki,alpha)
        flag1=0
        if order==1 and ki>0:
            flag1=1# sigma does not depend on k.
        if flag1==0:
            disp0=disp
            if (os.path.exists(dirSreal))==0:#calculate real axis using continuation
                brightness_factor=1
                # if MEMopt==1:
                #     om,sigrealA,sigrealB,sigoffdiag=calc_real_axis(U,T,order,k1,k2,k3,disp,kptnum,ki,alpha)
                # if MEMopt==2:
                #     om,sigrealA,sigrealB,sigoffdiag=calc_real_axis2(U,T,order,k1,k2,k3,disp,kptnum,ki,alpha)
                if MEMopt==1:#MEM for Mk
                    om,sigrealA,sigrealB,sigoffdiag=calc_real_axis(U,T,order,k1,k2,k3,0,0,0,alpha,0)#U,T,order,u,v,w,epsk,ttpts,indexofpt,alpha,ifsave=1
                elif MEMopt==-1:#pade for Mk, better use for order 1,2,3 but not 4.
                    om,sigrealA,sigrealB,sigoffdiag=calc_real_axis(U,T,order,k1,k2,k3,1,0,0,alpha,0)#U,T,order,u,v,w,epsk,ttpts,indexofpt,alpha,ifsave=1
                elif MEMopt==2:#MEM for sig
                    om,sigrealA,sigrealB,sigoffdiag=calc_real_axis2(U,T,order,k1,k2,k3,0,0,0,alpha,0)
                elif MEMopt==-2:# pade for sig
                    om,sigrealA,sigrealB,sigoffdiag=calc_real_axis2(U,T,order,k1,k2,k3,1,0,0,alpha,0)

            else:# read saved data
            
                Sfreal = np.loadtxt(dirSreal).T
                om=Sfreal[0,2:]#here we start from the 2nd line because the real axis self energy keeps first 2 lines from the imag axis self-energy...idk why but that is what i got.
                sigrealA=Sfreal[1,2:]+Sfreal[2,2:]*1j
                sigrealB=Sfreal[3,2:]+Sfreal[4,2:]*1j
                sigoffdiag=Sfreal[5,2:]+Sfreal[6,2:]*1j
            # plt.plot(om,sigrealA.real,label='sigA real')
            # plt.plot(om,sigrealA.imag,label='sigA imag')
            # plt.plot(om,sigrealB.real,label='sigB real')
            # plt.plot(om,sigrealB.imag,label='sigB imag') 
            # plt.legend()
            # plt.grid()
            # plt.show()

        # note:until now, the specturm function is even not defined on an even energy grid. I have to do the interpolation
        # to put it on an even grid.
        interpolated_sigA = np.zeros(len(energy_new))
        interpolated_sigB = np.zeros(len(energy_new))
        # interpolated_sigAB = np.zeros(len(energy_new))
        interpkind='cubic'
        interpolatorA = interp1d(om, sigrealA, kind=interpkind)
        interpolatorB = interp1d(om, sigrealB, kind=interpkind)
        interpolatedoff=interp1d(om, sigoffdiag, kind=interpkind)
        interpolated_sigA = interpolatorA(energy_new)
        interpolated_sigB = interpolatorB(energy_new)
        interpolated_off = interpolatedoff(energy_new)
        # if ifplot:
        #     plt.plot(energy_new,interpolated_sigA.real,label='Areal')
        #     plt.plot(energy_new,interpolated_sigA.imag,label='Aimag')
        #     plt.plot(energy_new,interpolated_sigB.real,label='Breal')
        #     plt.plot(energy_new,interpolated_sigB.imag,label='Bimag')      
        #     plt.legend()
        #     plt.title('interpolated sigreal')
        #     plt.show()
            #step4 real axis Green's function.
        GrealAB=(2*energy_new+2*mu-interpolated_sigA-interpolated_sigB)/((energy_new+mu-interpolated_sigB)*(energy_new+mu-interpolated_sigA)-disp**2)/2# GAA+GBB
        Grealoffdiag=2*(disp+interpolated_off)/((energy_new+mu-interpolated_sigB)*(energy_new+mu-interpolated_sigA)-disp**2)/2#
        # GrealAB=(2*energy_new+2*mu-interpolated_sigA-interpolated_sigB)/(energy_new+mu-interpolated_sigB)/(energy_new+mu-interpolated_sigA)/2# GAA+GBB
        # Grealoffdiag=2*(disp+interpolated_off)/(energy_new+mu-interpolated_sigB)/(energy_new+mu-interpolated_sigA)/2#
        # if flag1:
        #     GrealAB=(2*energy_new+2*mu-interpolated_sigA-interpolated_sigB)/((energy_new+mu-interpolated_sigB)*(energy_new+mu-interpolated_sigA)+disp0**2-disp**2)/2# GAA+GBB
        #     Grealoffdiag=2*(disp+interpolated_off)/((energy_new+mu-interpolated_sigB)*(energy_new+mu-interpolated_sigA)+disp0**2-disp**2)/2#            

        if debug:
            plt.plot(energy_new,GrealAB.real,label='diag real')
            plt.plot(energy_new,GrealAB.imag,label='diag imag')
            plt.plot(energy_new,Grealoffdiag.real,label='Grealoffdiag real')
            plt.plot(energy_new,Grealoffdiag.imag,label='Grealoffdiag imag')
            plt.legend()
            plt.title('Greal')
            plt.show()

        A_AB[:,ki]=-GrealAB.imag/np.pi
        A_off[:,ki]=-Grealoffdiag.imag/np.pi
        Attl[:,ki]=A_AB[:,ki]+A_off[:,ki]
        if debug:
            plt.plot(energy_new,Attl[:,ki],label='Attl')
            plt.axhline(y=0, color='r', linestyle='--', label='y=0')
            plt.legend()
            plt.title('Attl[:,ki] u={} v={} w={}'.format(k1,k2,k3))
            plt.show()        
        # time1=time.time()
        # print('time taken to get real-time G;',time1-time0)
        #plot specturm function
    energy=om
    if U<6:
        y_list=np.array([6,3,0,-3,-6])
        y_labels=['6','3','0','-3','-6']
    else:
        y_list=np.array([9,6,3,0,-3,-6,-9])
        y_labels=['9','6','3','0','-3','-6','-9']
    y_ticks=findyticks(energy_new,y_list)
    # print(y_ticks)
    


    if ifplot>0:
        vmin = np.percentile(Attl, 2) 
        vmax = np.percentile(Attl, 98) 
        ax = plt.gca()
        ax.imshow((Attl)*brightness_factor,aspect='auto', cmap='Blues',norm=colors.Normalize(vmin=vmin, vmax=vmax))
        ax.set_xticks(highsymindex)
        ax.set_xticklabels(klabels)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_ylabel('Energy/t')
        for tick in highsymindex:
            ax.axvline(x=tick, color='black', linestyle=':')
        ax.text(0.02, 0.98, 'N={}'.format(order), transform=ax.transAxes, 
        fontsize=20, color='black', va='top', ha='left')
        if ifplot==1:# show the plot
            plt.show()
        elif ifplot==2:# save the plot
            plt.savefig("../../paperwriting/raw_graphs/{}_{}_{}_{}_{}.png".format(U,T,order,alpha,MEMopt), dpi=500)

    return 0




def gen_DOS(U,T,order,alpha,MEMopt=1,knum=14,ifplot=0):
    '''
    To get dos I have to integrate over k. which is supposed to be a 10*10*10 grid. and another issue is try to find a symmetry to reduce the kpoints we need.
    In cartesian coordinate, kx,ky,kz are defined between -pi/a and pi/a.
    '''
    # knum=24
    mu=U/2
    energynum=1000
    sigma_min=0.3# gaussian smearing: all gaussian peaks will be replace as peaks with sigma=0.3
    maxpeak=0.5
    kxlist=(np.arange(knum)-knum/2+0.5)/knum*2# unit:pi/a
    kylist=(np.arange(knum)-knum/2+0.5)/knum*2
    kzlist=(np.arange(knum)-knum/2+0.5)/knum*2
    ifkdone=np.zeros((knum,knum,knum))
    Emin=-(6+U)
    Emax=(6+U)
    energynum=1000
    energy_new=np.linspace(Emax, Emin, num=energynum)
    Ak=np.zeros((energynum,knum,knum,knum))
    Ak_smeared=np.zeros((energynum,knum,knum,knum))
    flag1=0
    ttk=0
    for ikx,kx in enumerate(kxlist):
        for iky,ky in enumerate(kylist):
            for ikz,kz in enumerate(kzlist):   
                if order>=2 and MEMopt>0:                 
                    print(ikx,iky,ikz)
                if ikx+iky+ikz>0 and order==1 and flag1==0:
                    flag1=1
                    
                if ifkdone[ikx,iky,ikz]==0:# do it
                    ttk+=1
                    # convert cartesian coordinate into linear combination of reciprocal vectors
                    u=0.5*(ky+kz)
                    v=0.5*(kx+kz)
                    w=0.5*(kx+ky)
                    disp=lib.dispersion(u,v,w)
                    if flag1==0:
                        disp0=disp
                        if MEMopt==1:#MEM for Mk
                            om,sigrealA,sigrealB,sigoffdiag=calc_real_axis(U,T,order,u,v,w,0,0,0,alpha,0)#U,T,order,u,v,w,epsk,ttpts,indexofpt,alpha,ifsave=1
                        elif MEMopt==-1:#pade for Mk, better use for order 1,2,3 but not 4.
                            om,sigrealA,sigrealB,sigoffdiag=calc_real_axis(U,T,order,u,v,w,1,0,0,alpha,0)#U,T,order,u,v,w,epsk,ttpts,indexofpt,alpha,ifsave=1
                        elif MEMopt==2:#MEM for sig
                            om,sigrealA,sigrealB,sigoffdiag=calc_real_axis2(U,T,order,u,v,w,0,0,0,alpha,0)
                        elif MEMopt==-2:# pade for sig
                            om,sigrealA,sigrealB,sigoffdiag=calc_real_axis2(U,T,order,u,v,w,1,0,0,alpha,0)
                    interpolated_sigA = np.zeros(len(energy_new))
                    interpolated_sigB = np.zeros(len(energy_new))
                    # interpolated_sigAB = np.zeros(len(energy_new))
                    interpkind='cubic'
                    interpolatorA = interp1d(om, sigrealA, kind=interpkind)
                    interpolatorB = interp1d(om, sigrealB, kind=interpkind)
                    interpolatedoff=interp1d(om, sigoffdiag, kind=interpkind)
                    interpolated_sigA = interpolatorA(energy_new)
                    interpolated_sigB = interpolatorB(energy_new)
                    interpolated_off = interpolatedoff(energy_new)
                    # if MEMopt==1:# extrapolation of Mk
                    #     GrealAB=(2*energy_new+2*mu-interpolated_sigA-interpolated_sigB)/(energy_new+mu-interpolated_sigB)/(energy_new+mu-interpolated_sigA)/2# GAA+GBB
                    #     Grealoffdiag=2*(disp+interpolated_off)/(energy_new+mu-interpolated_sigB)/(energy_new+mu-interpolated_sigA)/2#
                    #     if flag1:
                    #         GrealAB=(2*energy_new+2*mu-interpolated_sigA-interpolated_sigB)/((energy_new+mu-interpolated_sigB)*(energy_new+mu-interpolated_sigA)+disp0**2-disp**2)/2# GAA+GBB
                    #         Grealoffdiag=2*(disp+interpolated_off)/((energy_new+mu-interpolated_sigB)*(energy_new+mu-interpolated_sigA)+disp0**2-disp**2)/2#   
                    # elif MEMopt==2:# extrapolation of Sigma
                    GrealAB=(2*energy_new+2*mu-interpolated_sigA-interpolated_sigB)/((energy_new+mu-interpolated_sigB)*(energy_new+mu-interpolated_sigA)-disp**2)/2# GAA+GBB
                    Grealoffdiag=2*(disp+interpolated_off)/((energy_new+mu-interpolated_sigB)*(energy_new+mu-interpolated_sigA)-disp**2)/2#
                        # if flag1:
                        #     GrealAB=(2*energy_new+2*mu-interpolated_sigA-interpolated_sigB)/((energy_new+mu-interpolated_sigB)*(energy_new+mu-interpolated_sigA)+disp0**2-disp**2)/2# GAA+GBB
                        #     Grealoffdiag=2*(disp+interpolated_off)/((energy_new+mu-interpolated_sigB)*(energy_new+mu-interpolated_sigA)+disp0**2-disp**2)/2#   

                    Ak[:,ikx,iky,ikz]=-GrealAB.imag/np.pi-Grealoffdiag.imag/np.pi

                    if np.max(Ak[:,ikx,iky,ikz])>maxpeak:# need some smearing
                        initial_guess = [np.max(Ak[:,ikx,iky,ikz]), energy_new[np.argmax(Ak[:,ikx,iky,ikz])], 0.2] 
                        popt, pcov = curve_fit(single_gaussian, energy_new, Ak[:,ikx,iky,ikz], p0=initial_guess)
                        rest=copy.deepcopy(Ak[:,ikx,iky,ikz])
                        A1, mu1, sigma1= popt
                        if sigma1<0:
                            sigma1=-sigma1
                        smearing1=sigma_min/sigma1
                        widthtimes1=min(smearing1*2,3)
                        E_window1=np.where((energy_new > mu1 - widthtimes1 * sigma1) & (energy_new < mu1 + widthtimes1 * sigma1))[0]
                        ori_int1=-scipy.integrate.trapz(Ak[E_window1,ikx,iky,ikz],energy_new[E_window1])# Note: E is from high to low!
                        # print('Sigma of Gauss1={}'.format(sigma1))
                        

                        A1=(ori_int1/np.sqrt(2*np.pi)/sigma1)# to keep the integration as the same. However, this should not change the amplitute of Gaussian too much.
                        if ifplot>=2:
                            print('A1={:.4f} mu1={:.4f} sigma1={:.4f}'.format(A1,mu1,sigma1))
                        gaussian_fitted=single_gaussian(energy_new, A1, mu1, sigma1)
                        rest[E_window1]=0

                        # diff=Ak[:,ikx,iky,ikz]-gaussian_fitted
                        allfit=gaussian_fitted
                        Ak_smeared[:,ikx,iky,ikz]=rest+single_gaussian(energy_new,A1/smearing1,mu1,sigma1*smearing1)

                        if np.max(rest)>maxpeak:# there is more than 1 peak to be processed.
                            initial_guess = [np.max(rest), energy_new[np.argmax(rest)], 0.2]
                            popt, pcov = curve_fit(single_gaussian, energy_new, rest, p0=initial_guess)
                            A2, mu2, sigma2 = popt
                            if sigma2<0:
                                sigma2=-sigma2
                            if ifplot>=2:
                                print('A2={:.4f} mu2={:.4f} sigma2={:.4f}'.format(A2,mu2,sigma2))
                            smearing2=sigma_min/sigma2
                            widthtimes2=min(smearing2*2,3)
                            E_window2=np.where((energy_new > mu2 - widthtimes2 * sigma2) & (energy_new < mu2 + widthtimes2 * sigma2))[0]
                            ori_int2=-scipy.integrate.trapz(rest[E_window2],energy_new[E_window2])
                            A2=(ori_int2/np.sqrt(2*np.pi)/sigma2)

                            # print('Sigma of Gauss2={}'.format(sigma2))
                            gaussian_fitted2=single_gaussian(energy_new, A2, mu2, sigma2)
                            allfit+=gaussian_fitted2
                            rest[E_window2]=0
                            Ak_smeared[:,ikx,iky,ikz]=rest+single_gaussian(energy_new,A2/smearing2,mu2,sigma2*smearing2)+single_gaussian(energy_new,A1/smearing1,mu1,sigma1*smearing1)

                            
                        # Ak_smeared[:,ikx,iky,ikz]=copy.deepcopy(smeared)
                    else:
                        Ak_smeared[:,ikx,iky,ikz]=copy.deepcopy(Ak[:,ikx,iky,ikz])
                    

                    if ifplot>=2:
                        if np.max(Ak[:,ikx,iky,ikz])>maxpeak:
                            plt.plot(energy_new,allfit,label='Gaussian fit')
                            plt.plot(energy_new,Ak_smeared[:,ikx,iky,ikz],label='Ak_smeared')
                            # plt.plot(energy_new,Ak[:,ikx,iky,ikz]-allfit,label='restmax={:.3f}'.format(np.max(np.abs(Ak[:,ikx,iky,ikz]-allfit))))
                        plt.plot(energy_new,Ak[:,ikx,iky,ikz],label='Attl')
                        # plt.axhline(y=0, color='r', linestyle='--', label='y=0')
                        plt.legend()
                        plt.title('Attl kx={} ky={} kz={}'.format(kx,ky,kz))
                        plt.show()  
                    ifkdone[ikx,iky,ikz]=1
                    # after finishing the first point in this symmetry, label all symmetry k points as done. and fill spectrum function for all of them.
                    for iikx in np.array([ikx,knum-1-ikx]):
                        for iiky in np.array([iky,knum-1-iky]):
                            for iikz in np.array([ikz,knum-1-ikz]):
                                Ak[:,iikx,iiky,iikz]=Ak[:,ikx,iky,ikz]#xyz
                                Ak_smeared[:,iikx,iiky,iikz]=Ak_smeared[:,ikx,iky,ikz]
                                ifkdone[iikx,iiky,iikz]=1

                                Ak[:,iikx,iikz,iiky]=Ak[:,ikx,iky,ikz]#xzy
                                Ak_smeared[:,iikx,iikz,iiky]=Ak_smeared[:,ikx,iky,ikz]
                                ifkdone[iikx,iikz,iiky]=1

                                Ak[:,iiky,iikz,iikx]=Ak[:,ikx,iky,ikz]#yzx
                                Ak_smeared[:,iiky,iikz,iikx]=Ak_smeared[:,ikx,iky,ikz]
                                ifkdone[iiky,iikz,iikx]=1

                                Ak[:,iiky,iikx,iikz]=Ak[:,ikx,iky,ikz]#yxz
                                Ak_smeared[:,iiky,iikx,iikz]=Ak_smeared[:,ikx,iky,ikz]
                                ifkdone[iiky,iikx,iikz]=1

                                Ak[:,iikz,iiky,iikx]=Ak[:,ikx,iky,ikz]#zyx
                                Ak_smeared[:,iikz,iiky,iikx]=Ak_smeared[:,ikx,iky,ikz]
                                ifkdone[iikz,iiky,iikx]=1

                                Ak[:,iikz,iikx,iiky]=Ak[:,ikx,iky,ikz]#zxy
                                Ak_smeared[:,iikz,iikx,iiky]=Ak_smeared[:,ikx,iky,ikz]
                                ifkdone[iikz,iikx,iiky]=1
    dos_raw=np.sum(Ak_smeared,axis=(1,2,3))/knum**3
    dos=copy.deepcopy(dos_raw)
    # dos[:-1]+=dos_raw[1:]
    # dos[:-2]+=dos_raw[2:]
    # dos[1:]+=dos_raw[:-1]
    # dos[2:]+=dos_raw[:-2]
    # dos[2:-2]/=5
    # dos[0]/=3
    # dos[-1]/=3
    # dos[1]/=4
    # dos[-2]/=4

    dos_ori=np.sum(Ak,axis=(1,2,3))/knum**3
    print('ttk%=',ttk/knum**3)
    # dosdir='./dosdata/DOS_{}_{}_{}_{}.txt'.format(U,T,order,alpha)
    dosdir='./dosdata/{}_{}/DOS_{}_{}_{}_{}_{}.txt'.format(U,T,U,T,order,MEMopt,alpha)
    dosfile=np.zeros((energynum,3))
    dosfile[:,0]=energy_new
    dosfile[:,1]=dos
    dosfile[:,2]=dos_ori
    np.savetxt(dosdir,dosfile)
    if ifplot>0:
        plt.plot(energy_new,dos,label='smeared')
        plt.plot(energy_new,dos_ori,label='original')
        plt.title('DOS: U={} T={} order={} alpha={}'.format(U,T,order,alpha))
        if ifplot==1:
            plt.show()

    return 0

def read_DOS(U,T,order,alpha,MEMopt):
    # dosdir='./dosdata/DOS_{}_{}_{}_{}.txt'.format(U,T,order,alpha)
    dosdir='./dosdata/{}_{}/DOS_{}_{}_{}_{}_{}.txt'.format(U,T,U,T,order,MEMopt,alpha)
    dosfile=np.loadtxt(dosdir)
    energy_new=dosfile[:,0]
    dos=dosfile[:,1]
    plt.plot(energy_new,dos)
    plt.title('DOS: U={} T={} order={} alpha={}'.format(U,T,order,alpha))
    plt.show()  
    return 0


def read_all_dos(U,T,alpha,MEMopt,maxorder=4,plotopt=1):
    colors = cm.inferno(np.linspace(0.2, 0.8, 5)[::-1])
    for order in np.arange(maxorder+2)-1:
        dosdir='./dosdata/{}_{}/DOS_{}_{}_{}_{}_{}.txt'.format(U,T,U,T,order,MEMopt,alpha)
        if (os.path.exists(dosdir))==1:
            dosfile=np.loadtxt(dosdir)
            energy_new=dosfile[:,0]
            dos=dosfile[:,1]
            if order>=0:
                lab='order {}'.format(order)
                plt.plot(energy_new,dos,color=colors[order],label=lab)
            elif order==-1:
                lab='DMFT'
                plt.plot(energy_new,dos,color='b',linestyle='--',label=lab)
        else:
            print('cannot find {}!'.format(dosdir))
    # plt.title('DOS: U={} T={} alpha={} MEM={}'.format(U,T,alpha,MEMopt))
    plt.legend()
    plt.xlim(-5,5)
    if plotopt==1:
        plt.show()
    elif plotopt==2:
        plt.savefig("../../paperwriting/raw_graphs/DOS_{}_{}_{}_{}.png".format(U,T,alpha,MEMopt), dpi=500)
    return 0


def read_all_dos_lowT(U,T,alpha,MEMopt,maxorder=4,plotopt=1):
    colors = cm.plasma(np.linspace(0, 0.8, 5)[::-1])#inferno
    for order in np.arange(maxorder+2)-1:
        # if order>=0:
        #     MEMopt=2# MEM, sigma
        # else:
        #     MEMopt=2# MEM,sigma
        dosdir='./dosdata/{}_{}/DOS_{}_{}_{}_{}_{}.txt'.format(U,T,U,T,order,MEMopt,alpha)
        if (os.path.exists(dosdir))==1:
            dosfile=np.loadtxt(dosdir)
            energy_new=dosfile[:,0]
            dos_raw=dosfile[:,1]


            dos = savgol_filter(dos_raw, window_length=15, polyorder=3)
            # dos=copy.deepcopy(dos_raw)
            # dos[:-1]+=dos_raw[1:]
            # dos[:-2]+=dos_raw[2:]
            # dos[1:]+=dos_raw[:-1]
            # dos[2:]+=dos_raw[:-2]
            # dos[2:-2]/=5
            # dos[0]/=3
            # dos[-1]/=3
            # dos[1]/=4
            # dos[-2]/=4


            if order>=0:
                lab='N={}'.format(order)
                mylinewidth=(order+3)/3
                # if order==4:
                #     mylinewidth=2
                plt.plot(energy_new,dos,color=colors[order],label=lab,linewidth=mylinewidth)
            elif order==-1:
                lab='DMFT'
                plt.plot(energy_new,dos,color='black',linestyle='--',label=lab)
        else:
            print('cannot find {}!'.format(dosdir))
    
    # DCAdirup="../../paperwriting/dos/DCA_8.0_0.24_UP.txt"
    # DCAdirdn="../../paperwriting/dos/DCA_8.0_0.24_DN.txt"
    # DCAup=np.loadtxt(DCAdirup,delimiter=',')
    # DCAdn=np.loadtxt(DCAdirdn,delimiter=',')
    # DCAupdos=interp1d(DCAup[:,1],DCAup[:,0],kind='linear', fill_value='extrapolate')
    # DCAdndos=interp1d(DCAdn[:,1],DCAdn[:,0],kind='linear', fill_value='extrapolate')
    # energylist=np.arange(1000)/1000*12
    # DCAdos=(DCAupdos(energylist)+DCAdndos(energylist))/24
    # plt.plot(energylist,DCAdos,color='grey',linestyle='--',label='DCA')

    # QMCdos=np.loadtxt("../../paperwriting/dos/QMC_8.0_0.25.txt",delimiter=',')
    # plt.plot(QMCdos[:,0],QMCdos[:,1],color='cyan',linestyle='--',label='QMC')

    # plt.title('DOS: U={} T={} alpha={} MEM={}'.format(U,T,alpha,MEMopt))
    plt.xlim(0,10)
    # plt.ylim(0,0.25)
    plt.xlabel('E/t')
    plt.ylabel('DOS')
    plt.legend()
    plt.tight_layout()
    if plotopt==1:
        plt.show()
    elif plotopt==2:
        plt.savefig("../../paperwriting/raw_graphs/DOS_{}_{}_{}_{}.png".format(U,T,alpha,MEMopt), dpi=1000)
    return 0


def gen_all_DOS(U,T,alpha,MEMopt,maxorder=4,knum=16):
    for order in np.arange(maxorder)+1:
        gen_DOS(U,T,order,alpha,MEMopt,knum)

def doscheck(U,T,alpha):
    for order in np.array([0,1]):
        for MEMopt in np.array([-1,1,2]):
            dosdir='./dosdata/{}_{}/DOS_{}_{}_{}_{}_{}.txt'.format(U,T,U,T,order,MEMopt,alpha)
            if (os.path.exists(dosdir))==1:
                dosfile=np.loadtxt(dosdir)
                energy_new=dosfile[:,0]
                dos=dosfile[:,1]
                plt.plot(energy_new,dos,label='order={} MEMopt={}'.format(order, MEMopt))
    plt.title('DOS: U={} T={} alpha={}'.format(U,T,order,alpha))
    plt.legend()
    plt.show()  
    return 0

def doscheck_mem(U,T,alpha,order):

    

    for MEMopt in np.array([-2,2]):#,-1,1
        if MEMopt==-1:
            lab='pade of M'
        elif MEMopt==1:
            lab='MEM of M'
        elif MEMopt==2:
            lab='MEM of Sigma'
        elif MEMopt==-2:
            lab='pade of Sigma'
        if order<=0:
            specplot_run.gen_DOS(U,T,order,alpha,MEMopt,0)
        else:
            gen_DOS(U,T,order,alpha,MEMopt)
        dosdir='./dosdata/{}_{}/DOS_{}_{}_{}_{}_{}.txt'.format(U,T,U,T,order,MEMopt,alpha)
        if (os.path.exists(dosdir))==1:
            dosfile=np.loadtxt(dosdir)
            energy_new=dosfile[:,0]
            dos=dosfile[:,1]
            plt.plot(energy_new,dos,label='order={} {}'.format(order, lab))
    plt.title('DOS: U={} T={} alpha={}'.format(U,T,order,alpha))
    plt.legend()
    plt.show()  
    return 0


if __name__ == "__main__":
    sizefont=16# default:12
    plt.rc('font', size=sizefont) 
    plt.rc('axes', titlesize=sizefont) 
    plt.rc('axes', labelsize=sizefont) 
    plt.rc('xtick', labelsize=sizefont)
    plt.rc('ytick', labelsize=sizefont)
    plt.rc('legend', fontsize=13)
    #default settings
    time0=time.time()
    numfiles=5# take last 10 sigma files
    order=3#perturbation order
    T=0.25
    U=8.
    outputdir='./spec_data/{}_{}/'.format(U,T)
    MEM=2# 1=M=1/(iom-(Sig(iom)-Sig(inf))), 2=sigma if with a - sign, which means use pade instead of MEM
    alpha=0.3
    alpha2=0.3

    gamma=0.001
    Norder=100
    # gen_spec_plot(8.0,0.41,1,alpha2,MEM,1,2)
    # gen_spec_plot(8.0,0.41,2,alpha2,MEM,1,2)
    # gen_spec_plot(8.0,0.41,3,alpha2,MEM,1,2)    
    # gen_spec_plot(8.0,0.41,4,alpha2,MEM,1,2)

    # gen_spec_plot(8.0,0.25,1,alpha2,MEM,1,2)
    # gen_spec_plot(8.0,0.25,2,alpha2,MEM,1,2)
    # gen_spec_plot(8.0,0.25,3,alpha2,MEM,1,2)    
    # gen_spec_plot(8.0,0.25,4,alpha2,MEM,1,2)


    # specplot_run.gen_DOS(8.0,0.41,-1,alpha2,MEM,1)
    # specplot_run.gen_DOS(8.0,0.41,0,alpha2,MEM,1)
    # gen_DOS(8.0,0.41,1,alpha2,MEM)
    # gen_DOS(8.0,0.41,2,alpha2,MEM)
    # gen_DOS(8.0,0.41,3,alpha2,MEM)
    # gen_DOS(8.0,0.41,4,alpha2,MEM)


    # specplot_run.gen_DOS(8.0,0.25,-1,alpha2,MEM,1)
    # specplot_run.gen_DOS(8.0,0.25,0,alpha2,MEM,1)
    # gen_DOS(8.0,0.25,1,alpha2,MEM,14,0)
    # gen_DOS(8.0,0.25,2,alpha2,MEM,14,0)
    # gen_DOS(8.0,0.25,3,alpha2,MEM,14,0)
    # gen_DOS(8.0,0.25,4,alpha2,MEM,14,0)
    # gen_DOS(8.0,0.25,2,alpha,MEM,14,0)
    # gen_DOS(8.0,0.25,3,alpha,MEM,14,0)
    # gen_DOS(8.0,0.25,4,alpha,MEM,14,0)
    
    # gen_spec_plot(8.0,0.25,4,alpha2,MEM,1,2)
    # gen_spec_plot(8.0,0.25,3,alpha2,MEM,1,2)
    # gen_spec_plot(8.0,0.25,2,alpha2,MEM,1,2)    
    # gen_spec_plot(8.0,0.25,1,alpha2,MEM,1,2)



    # read_DOS(8.0,0.41,1,alpha2)
    # read_all_dos(8.0,0.41,alpha2,MEM,4)
    read_all_dos_lowT(8.0,0.41,alpha,MEM,4,2)
    # doscheck(U,T,alpha2)
    # doscheck_mem(8.0,0.25,alpha2,1)


    