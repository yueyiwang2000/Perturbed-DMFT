import  os,sys,subprocess
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../')
import perturb_lib as lib
import scipy
from scipy.interpolate import interp1d
from matplotlib.colors import LightSource
import matplotlib.colors as colors
import time
from scipy.optimize import curve_fit
import copy
from maxentropy import Pade
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
fileSig='Sig.out'
fileaveSig='sig.inpx'
filemodSig='sigmod.inpx'
filerealSig='Sig.out'



def triple_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3):
    g1 = A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
    g2 = A2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2))
    g3 = A3 * np.exp(-(x - mu3)**2 / (2 * sigma3**2))
    return g1 + g2 + g3

def double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
    g1 = A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
    g2 = A2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2))
    return g1 + g2

def single_gaussian(x, A1, mu1, sigma1):
    g1 = A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
    return g1

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

def gen_kbasis_new(Nmax,knum=10):
    '''
    basis based on Fourier basis.  because of the symmetry, we have:
    cos(n1*kx)cos(n2*ky)cos(n3*kz) as basis. ni=0,1,2,....Nmax-1.
    '''
    kbasis=np.zeros((2,Nmax**3,knum,knum,knum))
    for nx in np.arange(Nmax):
        for ny in np.arange(Nmax):
            for nz in np.arange(Nmax):
                kbasis[(nx+ny+nz)%2,nx*Nmax**2+ny*Nmax+nz,:,:,:]=Fourier_Basis(nx,ny,nz)
    return kbasis

def findmaxindex(directory):
    i=0 #start to check from Sig.out.1
    isfind=1
    while isfind==1:
        i+=1
        filename=directory+fileSig+'.{}'.format(i)
        isfind=os.path.isfile(filename)
    if i==1:
        print('Error: cannot find any Sig.out!')
    return i-1

def dist_kpoints(k1,k2):
    k1x=-k1[0]+k1[1]+k1[2]
    k1y=k1[0]-k1[1]+k1[2]
    k1z=k1[0]+k1[1]-k1[2]
    k2x=-k2[0]+k2[1]+k2[2]
    k2y=k2[0]-k2[1]+k2[2]
    k2z=k2[0]+k2[1]-k2[2]
    dist=np.sqrt((k1x-k2x)**2+(k1y-k2y)**2+(k1z-k2z)**2)
    return dist

def gen_entire_kpath(highsym_path,kpoints_per_dist=400):
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

def calc_real_axis(U,T,order=-1,MEMopt=1,alpha=0.15):
    '''
    order==-1 means DMFT
    order=0 means 0th order.
    MEMopt=1: using Mk.
    MEMopt=2: using sigma for MEM.
    '''
    outputdir='./spec_data/{}_{}/'.format(U,T)
    numfiles=5
    # for DMFT
    mu=U/2
    if U>=8:
        mode=0
    else:
        mode=1
    if mode==0:
        directory='../../files_boldc/{}_{}/'.format(U,T)
    elif mode==1:
        directory='../../files_ctqmc/{}_{}/'.format(U,T)
    else:
        print('Error: cannot read mode!')
    #step1: average the self-energies
    maxind=findmaxindex(directory)
    if maxind<5:
        print('Error: At least need 5 Sigma.out files!')
        return 0
    print('# of Sigma files found: ',maxind)
    #put all names of self-energies to be averaged all together
    allnames=' '
    for i in np.arange(numfiles):
        allnames=allnames+directory+fileSig+'.{} '.format(maxind-i)
    cmd_ave='python saverage3.py '+allnames
    # print(cmd_ave)
    subprocess.call(cmd_ave, shell=True)

 

    #step1.5: extract the sig_inf
    Sf = np.loadtxt(fileaveSig).T
    om=Sf[0,:]



    epsilon=0.01

    if order==-1:
        sigA=(Sf[1]-Sf[3])/2+mu  +(Sf[2]+Sf[4])/2*1j
        sigB=-(Sf[1]-Sf[3])/2+mu+(Sf[2]+Sf[4])/2*1j
        sigA_inf=sigA[-1].real
        sigB_inf=sigB[-1].real
    if order==0:
        sigA_inf=alpha*U/2+mu-epsilon
        sigB_inf=-alpha*U/2+mu+epsilon
    # elif order==0:
        sigA=alpha*U/2+mu+Sf[2]*1j
        sigB=-alpha*U/2+mu+Sf[4]*1j   


    if np.abs(MEMopt)==2:# using sigma
        f = open(filemodSig, 'w')
        for i,iom in enumerate(om):
            if order==-1:#DMFT
                print(iom, Sf[1,i]-sigA_inf, Sf[2,i], Sf[3,i]-sigB_inf,Sf[4,i], file=f) 
            elif order==0:# 0th order, replace the real part with a const splitting.
                print(iom, epsilon, Sf[2,i], -epsilon, Sf[4,i], file=f) # seems the maxent script only works when we put a tiny nonzero number there. don't know why.
        f.close()
        
        if MEMopt==2:
            #step2: run maxent and put sig_inf back
            cmd_maxent='python maxent_run.py '+filemodSig
            subprocess.call(cmd_maxent,shell=True)
            Sfreal = np.loadtxt(filerealSig).T
            omreal=Sfreal[0,2:]#here we start from the 2nd line because the real axis self energy keeps first 2 lines from the imag axis self-energy...idk why but that is what i got.
            sigrealA=Sfreal[1,2:]+sigA_inf+Sfreal[2,2:]*1j
            sigrealB=Sfreal[3,2:]+sigB_inf+Sfreal[4,2:]*1j
            if order==0:
                sigrealB=Sfreal[3,2:]+sigB_inf+Sfreal[2,2:]*1j
        elif MEMopt==-2:# pade of sigma
            omreal = np.linspace(-15,15,1001)
            # gamma=0.001
            # Norder=100
            if order==-1:
                sigrealA=Pade(om,sigA-sigA_inf,omreal,gamma,Norder)+sigA_inf
                sigrealB=Pade(om,sigB-sigB_inf,omreal,gamma,Norder)+sigB_inf
            elif order==0:
                sigrealA=Pade(om,sigA.imag*1j,omreal,gamma,Norder)+alpha*U/2+mu
                sigrealB=Pade(om,sigB.imag*1j,omreal,gamma,Norder)-alpha*U/2+mu


    elif np.abs(MEMopt)==1:# using Mk
        if order==-1:
            Mk1=1/(1j*om-(sigA-sigA_inf))
            Mk2=1/(1j*om-(sigB-sigB_inf))
        if order==0:
            Mk1=1/(1j*om-(Sf[2]*1j))
            Mk2=1/(1j*om-(Sf[4]*1j))
        f = open(filemodSig, 'w')
        for i,iom in enumerate(om):
            print(iom, Mk1[i].real, Mk1[i].imag, Mk2[i].real,Mk2[i].imag, file=f) 
        f.close()
        if MEMopt==1:# MEM for Mk
            cmd_maxent='python maxent_run.py '+filemodSig
            subprocess.call(cmd_maxent,shell=True)
            Sfreal = np.loadtxt(filerealSig).T
            omreal=Sfreal[0,2:]
            Mk1real=Sfreal[1,2:]+Sfreal[2,2:]*1j
            Mk2real=Sfreal[3,2:]+Sfreal[4,2:]*1j
            if order==0:
                Mk2real=Sfreal[1,2:]+Sfreal[2,2:]*1j

        elif MEMopt==-1:# pade for Mk
            omreal = np.linspace(-15,15,1001)

            Mk1real=Pade(om,Mk1,omreal,gamma,Norder)
            Mk2real=Pade(om,Mk2,omreal,gamma,Norder)
        # plt.scatter(om,Mk1.real,label='mk1.real')
        # plt.scatter(om,Mk1.imag,label='mk1.imag')
        # plt.plot(omreal,Mk1real.real,label='mk1real.real')
        # plt.plot(omreal,Mk1real.imag,label='mk1real.imag')
        # plt.legend()
        # plt.xlim(-15,15)
        # plt.show()
        # plt.scatter(om,Mk2.real,label='mk2.real')
        # plt.scatter(om,Mk2.imag,label='mk2.imag')
        # plt.plot(omreal,Mk2real.real,label='mk2real.real')
        # plt.plot(omreal,Mk2real.imag,label='mk2real.imag')
        # plt.legend()
        # plt.xlim(-15,15)
        # plt.show()
        if order==-1:
            sigrealA=omreal-1/Mk1real+sigA_inf
            sigrealB=omreal-1/Mk2real+sigB_inf
        if order==0:
            sigrealA=omreal-1/Mk1real+alpha*U/2+mu
            sigrealB=omreal-1/Mk2real-alpha*U/2+mu
    os.makedirs(outputdir, exist_ok=True)
    f = open(outputdir+'{}_{}_{}_{}_{}.dat'.format(U,T,order,MEMopt,alpha), 'w')
    for i,iom in enumerate(omreal):
        print(iom, sigrealA[i].real, sigrealA[i].imag, sigrealB[i].real, sigrealB[i].imag, file=f) 
    f.close()

    # try to plot the real axis self-energy

    return omreal,sigrealA,sigrealB#,om,sigA,sigB




def gen_spec_plot(U,T,order,alpha,MEMopt,ifplot=1):

        mu=U/2
        beta=1/T    
        brightness_factor=1
        omreal,sigrealA,sigrealB=calc_real_axis(U,T,order,MEMopt,alpha)   
        kptperdis=100

        # if(len(sys.argv)==5):#calculate real axis using continuation
        #     brightness_factor=1
        #     omreal,sigrealA,sigrealB,om,sigA,sigB=calc_real_axis(U,T,mode,myorder)


        # else:# read saved data
        #     brightness_factor = float(sys.argv[5])
        #     Sfreal = np.loadtxt(outputdir+'{}_{}_{}.dat'.format(myorder,U,T)).T
        #     omreal=Sfreal[0,2:]#here we start from the 2nd line because the real axis self energy keeps first 2 lines from the imag axis self-energy...idk why but that is what i got.
        #     sigrealA=Sfreal[1,2:]+Sfreal[2,2:]*1j
        #     sigrealB=Sfreal[3,2:]+Sfreal[4,2:]*1j
        # plt.plot(omreal,sigrealA.real,label='sigrealA real')
        # plt.plot(omreal,sigrealA.imag,label='sigrealA imag')
        # plt.scatter(om,sigA.real,label='sigA real')
        # plt.scatter(om,sigA.imag,label='sigA imag')
        # plt.legend()
        # plt.xlim(-20,20)
        # plt.grid()
        # plt.show()

        # plt.plot(omreal,sigrealB.real,label='sigrealB real')
        # plt.plot(omreal,sigrealB.imag,label='sigrealB imag') 
        # plt.scatter(om,sigB.real,label='sigB real')
        # plt.scatter(om,sigB.imag,label='sigB imag')
        # plt.legend()
        # plt.xlim(-20,20)
        # plt.grid()
        # plt.show()

        # note:until now, the specturm function is even not defined on an even energy grid. I have to do the interpolation
        # to put it on an even grid.

        #step2.5: interpolation of self-energy
        # Emin=-(6+U/2)
        # Emax=(6+U/2)
        Emin=-(6+U/2)
        Emax=(6+U/2)
        energy_new=np.linspace(Emax, Emin, num=1000)
        interpolated_sigA = np.zeros(len(energy_new))
        interpolated_sigB = np.zeros(len(energy_new))
        interpkind='cubic'
        interpolatorA = interp1d(omreal, sigrealA, kind=interpkind)
        interpolatorB = interp1d(omreal, sigrealB, kind=interpkind)
        interpolated_sigA = interpolatorA(energy_new)
        interpolated_sigB = interpolatorB(energy_new)

        plt.plot(energy_new,interpolated_sigA.real,label='Areal')
        plt.plot(energy_new,interpolated_sigA.imag,label='Aimag')
        plt.plot(energy_new,interpolated_sigB.real,label='Breal')
        plt.plot(energy_new,interpolated_sigB.imag,label='Bimag')
        plt.legend()
        plt.title('interpolated sigreal')
        plt.show()


        # step3: define a k path
        # Our lattice is FCC, reciprocal lattice is BCC. check this: https://lampx.tugraz.at/~hadley/ss1/bzones/bcc.php
        # point convention:k=n1k1+n2k2+n3k3.  we can call the library in python_src. 

        # #this is the reduced BZ, not unfolded. ГXWKГL UWLK. THIS IS THE MOST REASONABLE PATH. 
        # k_highsym_path=np.array([[0,0,0],#Г
        #                         [0,1/2,1/2],#X
        #                         [1/4,3/4,1/2],#W
        #                         [3/8,3/4,3/8],#k
        #                         [0,0,0],#Г
        #                         [1/2,1/2,1/2]#L
        #                         ])#http://lampx.tugraz.at/~hadley/ss1/bzones/fcc.php
        # #[1/4,5/8,5/8],#U
        
        # kpath,highsymindex=gen_entire_kpath(k_highsym_path)
        # k1=kpath[:,0].T
        # k2=kpath[:,1].T
        # k3=kpath[:,2].T
        # disp=lib.dispersion(k1,k2,k3)
        # klabels=['Г', 'X', 'W', 'K', 'Г', 'L']


        # however, to unfold, we need to put the dispersion in the BZ of original single unit cell. The kpath is:ГXMГRX
        #note: the reciprocal vector: k1=(-1,1,1)pi/a k2=(1,-1,1)pi/a k3=(1,1,-1)pi/a
        k_highsym_path=np.array([[0,0,0],#Г
                                [0,1/2,1/2],#X:kx=pi/a, ky=kz=0
                                [1/2,1/2,1],#M kx=ky=pi/a, kz=0
                                [0,0,0],#Г
                                [1,1,1],#R kx=ky=kz=pi/a
                                [0,1/2,1/2]#X
                                ])        
        kpath,highsymindex=gen_entire_kpath(k_highsym_path,kptperdis)
        k1=kpath[:,0].T
        k2=kpath[:,1].T
        k3=kpath[:,2].T
        disp=lib.dispersion(k1,k2,k3)
        klabels=['Г', 'X', 'M', 'Г', 'R',  'X']


        # plot the dispersion
        plt.plot(disp,color='blue')
        # plt.plot(-disp,color='blue')
        ax = plt.gca()
        ax.set_xticks(highsymindex)
        ax.set_xticklabels(klabels)
        ax.set_ylabel('Energy/t')
        for tick in highsymindex:
            ax.axvline(x=tick, color='black', linestyle=':')
        plt.show()




        #step4 real axis Green's function.
        # Note: usually we say A~Im(G). And off-diagonal G in this case, which is actually real, so no imag part, which is...reasonable? 
        # zA=om+mu-sigrealA
        # zB=om+mu-sigrealB
        zA=energy_new+mu-interpolated_sigA
        zB=energy_new+mu-interpolated_sigB

        GrealA=zB[:,None]/(zB[:,None]*zA[:,None]-disp[None,:]**2)
        GrealB=zA[:,None]/(zB[:,None]*zA[:,None]-disp[None,:]**2)
        Greal12=disp[None,:]/(zB[:,None]*zA[:,None]-disp[None,:]**2)
        A_A=-GrealA.imag/np.pi
        A_B=-GrealB.imag/np.pi#energy,k
        A_12=-Greal12.imag/np.pi
        Attl=(A_A+A_B+2*A_12)/2


        # energy=om
        if U<6:
            y_list=np.array([6,3,0,-3,-6])
            y_labels=['6','3','0','-3','-6']
        else:
            y_list=np.array([9,6,3,0,-3,-6,-9])
            y_labels=['9','6','3','0','-3','-6','-9']
        y_ticks=findyticks(energy_new,y_list)
    

        # TOTAL SPECTRA
        if ifplot>0:
            vmin = np.percentile(Attl, 2) 
            vmax = np.percentile(Attl, 98) 
            ax = plt.gca()
            ax.imshow(Attl*brightness_factor,aspect='auto', cmap='Blues',norm=colors.Normalize(vmin=vmin, vmax= vmax))
            ax.set_xticks(highsymindex)
            ax.set_xticklabels(klabels)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels)
            ax.set_ylabel('Energy/t')
            # ax.set_title('Spectral function: U={} T={}'.format(U,T))
            for tick in highsymindex:
                ax.axvline(x=tick, color='black', linestyle=':')
            if order==-1:
                mytext='DMFT'
            elif order==0:
                mytext='N=0'
            ax.text(0.02, 0.98, mytext, transform=ax.transAxes, 
            fontsize=20, color='black', va='top', ha='left')
            if ifplot==1:
                plt.show()
            elif ifplot==2:
                plt.savefig("../../paperwriting/raw_graphs/{}_{}_{}_{}_{}.png".format(U,T,order,alpha,MEMopt), dpi=500)

def gen_DOS(U,T,order,alpha,MEMopt,ifredo=0,ifplot=0):
    mu=U/2
    beta=1/T    
    outputdir='./spec_data/{}_{}/'.format(U,T)
    dir=outputdir+'{}_{}_{}_{}_{}.dat'.format(U,T,order,MEMopt,alpha)
    if (os.path.exists(dir))==0 or ifredo:
        omreal,sigrealA,sigrealB=calc_real_axis(U,T,order,MEMopt,alpha) 
    else:
        sigfile=np.loadtxt(dir)
        omreal=sigfile[:,0]
        sigrealA=sigfile[:,1]+1j*sigfile[:,2]
        sigrealB=sigfile[:,3]+1j*sigfile[:,4]
    # ifplot=2
    # smearing=5# increase sigma to 3 times
    sigma_min=0.3
    maxpeak=0.6
    energynum=1000
    knum=14
    kxlist=(np.arange(knum)-knum/2+0.5)/knum*2# unit:pi/a
    kylist=(np.arange(knum)-knum/2+0.5)/knum*2
    kzlist=(np.arange(knum)-knum/2+0.5)/knum*2
    ifkdone=np.zeros((knum,knum,knum))
    Emin=-(6+U)
    Emax=(6+U)
    energy_new=np.linspace(Emax, Emin, num=energynum)
    Ak=np.zeros((energynum,knum,knum,knum))
    Ak_smeared=np.zeros((energynum,knum,knum,knum))
    interpolated_sigA = np.zeros(len(energy_new))
    interpolated_sigB = np.zeros(len(energy_new))
    interpkind='cubic'
    interpolatorA = interp1d(omreal, sigrealA, kind=interpkind)
    interpolatorB = interp1d(omreal, sigrealB, kind=interpkind)
    interpolated_sigA = interpolatorA(energy_new)
    interpolated_sigB = interpolatorB(energy_new)
    for ikx,kx in enumerate(kxlist):
        for iky,ky in enumerate(kylist):
            for ikz,kz in enumerate(kzlist):                    

                    
                if ifkdone[ikx,iky,ikz]==0:# do it
                    # convert cartesian coordinate into linear combination of reciprocal vectors
                    u=0.5*(ky+kz)
                    v=0.5*(kx+kz)
                    w=0.5*(kx+ky)
                    disp=lib.dispersion(u,v,w)

                    GrealAB=(2*energy_new+2*mu-interpolated_sigA-interpolated_sigB)/((energy_new+mu-interpolated_sigB)*(energy_new+mu-interpolated_sigA)-disp**2)/2# GAA+GBB
                    Grealoffdiag=2*(disp)/((energy_new+mu-interpolated_sigB)*(energy_new+mu-interpolated_sigA)-disp**2)/2#

                    Ak[:,ikx,iky,ikz]=-GrealAB.imag/np.pi-Grealoffdiag.imag/np.pi



                    if np.max(Ak[:,ikx,iky,ikz])>maxpeak:# need some smearing
                        # plt.plot(energy_new,Ak[:,ikx,iky,ikz],label='Attl')
                        # plt.show()
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
                        if np.max(Ak[:,ikx,iky,ikz])>0.8:
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
                                Ak_smeared[:,iikx,iiky,iikz]=Ak_smeared[:,ikx,iky,ikz]#xyz
                                Ak[:,iikx,iiky,iikz]=Ak[:,ikx,iky,ikz]
                                ifkdone[iikx,iiky,iikz]=1

                                Ak_smeared[:,iikx,iikz,iiky]=Ak_smeared[:,ikx,iky,ikz]#xzy
                                Ak[:,iikx,iikz,iiky]=Ak[:,ikx,iky,ikz]
                                ifkdone[iikx,iikz,iiky]=1

                                Ak_smeared[:,iiky,iikz,iikx]=Ak_smeared[:,ikx,iky,ikz]#yzx
                                Ak[:,iiky,iikz,iikx]=Ak[:,ikx,iky,ikz]
                                ifkdone[iiky,iikz,iikx]=1

                                Ak_smeared[:,iiky,iikx,iikz]=Ak_smeared[:,ikx,iky,ikz]#yxz
                                Ak[:,iiky,iikx,iikz]=Ak[:,ikx,iky,ikz]
                                ifkdone[iiky,iikx,iikz]=1

                                Ak_smeared[:,iikz,iiky,iikx]=Ak_smeared[:,ikx,iky,ikz]#zyx
                                Ak[:,iikz,iiky,iikx]=Ak[:,ikx,iky,ikz]
                                ifkdone[iikz,iiky,iikx]=1

                                Ak_smeared[:,iikz,iikx,iiky]=Ak_smeared[:,ikx,iky,ikz]#zxy
                                Ak[:,iikz,iikx,iiky]=Ak[:,ikx,iky,ikz]
                                ifkdone[iikz,iikx,iiky]=1
    dos=np.sum(Ak_smeared,axis=(1,2,3))/knum**3
    dos_ori=np.sum(Ak,axis=(1,2,3))/knum**3
    dosdir='./dosdata/{}_{}/DOS_{}_{}_{}_{}_{}.txt'.format(U,T,U,T,order,MEMopt,alpha)
    dosfile=np.zeros((energynum,3))
    dosfile[:,0]=energy_new
    dosfile[:,1]=dos[::-1]
    dosfile[:,2]=dos_ori
    os.makedirs('./dosdata/{}_{}/'.format(U,T), exist_ok=True)
    np.savetxt(dosdir,dosfile)
    plt.plot(energy_new,dos,label='smeared')
    plt.plot(energy_new,dos_ori,label='original')
    plt.legend()
    plt.title('DOS: U={} T={} order={} alpha={} MEM={}'.format(U,T,order,alpha,MEMopt))
    plt.show()


    return 0

def read_DOS(U,T,order,alpha,MEMopt):
    dosdir='./dosdata/{}_{}/DOS_{}_{}_{}_{}_{}.txt'.format(U,T,U,T,order,MEMopt,alpha)
    dosfile=np.loadtxt(dosdir)
    energy_new=dosfile[:,0]
    dos=dosfile[:,1]
    plt.plot(energy_new,dos)
    plt.title('DOS: U={} T={} order={} alpha={}'.format(U,T,order,alpha))
    plt.show()  
    return 0




if __name__ == "__main__":
    #default settings
    sizefont=16# default:12
    plt.rc('font', size=sizefont) 
    plt.rc('axes', titlesize=sizefont) 
    plt.rc('axes', labelsize=sizefont) 
    plt.rc('xtick', labelsize=sizefont)
    plt.rc('ytick', labelsize=sizefont)
    plt.rc('legend', fontsize=13)
    order=0#perturbation order
    T=0.41
    U=8.0
    order=-1# -1 means DMFT, 0 means 0th order
    MEM=2# 1=Mk, 2=sigma, -1=Mkpade, -2=sigmapade
    alpha=0.3
    gamma=0.001
    Norder=100
    # mode=0#0=boldc, 1=ctqmc
    # gen_spec_plot(U,T,-1,alpha,MEM,2)
    # gen_spec_plot(U,T,0,alpha,MEM,2)
    gen_DOS(U,T,-1,alpha,MEM,1,0)
    gen_DOS(U,T,0,alpha,MEM,1,0)
    # read_DOS(U,T,order,alpha)
    # doscheck(U,T,alpha)