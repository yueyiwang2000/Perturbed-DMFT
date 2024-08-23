# @Copyright 2024 Yueyi Wang
import numpy as np
import matplotlib.pyplot as plt 
import perturb_lib as lib
import perturb_imp as libimp
import fft_convolution as libfft
import diagrams
import sys,os,subprocess
'''
This code aims to calculate the Free energy (and also maybe energy) of DMFT+perturbation. For theory part, check note 'Free energy in DMFT'
The algorithm is: Gamma[G]=TrlogG-Tr(SigmaG)+Phi[G]. 
For first 2 terms we just plug in the G, the best avaliable GF, which is DMFT+perturbation, and corresponded Sigma.
The modification for Phi is:
Phi_best=\sum_n (Phi(n)[Gbest]-Phi(n)[Gimp])+Phi[Gimp].
Here Phi(n)[G] is all nth order diagrams of Phi constructed by G. And n runs from 1 to as high order as we can. (currently up to 3)
Phi[Gimp] is given by the imp solver, from the python scripts in Tools/FreeEnergy, cmpEimp_ctqmc.py (for ctqmc) and cmpEimp_bold.py (for bold).

How to use this script: since it is hard to save a k-dependent G and Sigma, this file works like a library, which provides a function to calculate energies.
'''


def readDMFT(dir,nfreq): # read DMFT Sigma and G.
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
            if (os.path.exists(filename)):
                filefound=1
                break
            if i<10:
                print('warning: only {} DMFT iterations. result might not be accurate!'.format(i))
                break

    if filefound==0:
        print('{} cannot be found!'.format(filename))  
    # else:
    #     print('reading DMFT data from {}'.format(filename))



    sigma=np.loadtxt(filename)[:nfreq,:]
    sigA=sigma[:,1]+1j*sigma[:,2]
    sigB=sigma[:,3]+1j*sigma[:,4]
    return sigA,sigB# this also works for G!

def gen_energy_files(U,T,opt1=0):
    '''
    run the calculation of Eimp, Fimp and Phi. Here I store the in the same dir as the DMFT output, because if i have to redo DMFT the old file will be cleaned.
    '''
    if U>=8.:
        opt=0
    else:
        opt=1
    # U <=5 ctqmc is used, otherwize boldc.
    if opt==0:
        dire='../files_boldc/{}_{}/'.format(U,T)
        cmd='python ../Tools/FreeEnergy/cmpEimp_bold.py {} {} {} > ../perturbation/impenergydata/{}_{}.txt'.format(U,T,dire,U,T)
        subprocess.call(cmd, shell=True)
    else:
        dire='../files_ctqmc/{}_{}/'.format(U,T)
        cmd='python ../Tools/FreeEnergy/cmpEimp_ctqmc.py {} {} {} > ../perturbation/impenergydata/{}_{}.txt'.format(U,T,dire,U,T)
        subprocess.call(cmd, shell=True)
    return 0

def find_last_value(file_path,name):
    """
    Search for the last occurrence of 'name' in a file and return the float value following it.
    
    :param file_path: Path to the file to be searched.
    :return: The float value following the last occurrence of 'Fimp=' or None if not found.
    """
    last_fimp_value = None
    if os.path.exists(file_path)==False:
        print('cannnot find the file {} !'.format(file_path))
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if name in line:
                    parts = line.split()
                    for part in parts:
                        if part.startswith(name):
                            # Try to extract and convert the number following 'name'
                            try:
                                last_fimp_value = float(part.split('=')[1])
                            except ValueError:
                                # If conversion fails, continue to the next occurrence
                                continue
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    # print(last_fimp_value)
    return last_fimp_value

def ferm(x):
    # Create an array from x if it's not already an array
    x = np.array(x)
    
    # Initialize the result array with the same shape as x
    result = np.zeros_like(x)
    
    # Boolean masks for the extreme values
    mask_large = x > 300
    mask_small = x < -300

    # Handle the extreme values
    result[mask_large] = 0
    result[mask_small] = 1

    # Handle the non-extreme values
    mask_normal = ~np.logical_or(mask_large, mask_small)
    result[mask_normal] = 1 / (np.exp(x[mask_normal]) + 1)

    return result

def GetHighFrequency(CC,om):
    " Approximates CC ~  A/(i*om-C) "
    A = 1./(1/(CC[-1]*om[-1])).imag
    C = -A*(1./CC[-1]).real# before taking real part this C is purely imaginary.
    return (A, C)

def ReturnHighFrequency(A,C,beta,E):
    " Returns the value of Tr(A/(iom-C) 1/(iom-E) )"
    return A*(ferm(E*beta)-ferm(C*beta))/(E-C)

def fTrSigmaG(om, Gfc, Sigc, EimpS, beta,knum=10):
    '''
    TrsigmaG is used in many places in this code.
    The Baym-Kadanoff functional directly contains this term.
    Also, the Phi is some sort of this term: a self-energy diagram is closed by a G to get a term in Phi.
    Generally, I want this can be used for both 1D (impurity) and 4D (dispersive) case.
    '''
    SDf=0.0
    # print('TrSigmaG: Gfc.shape=',np.shape(Gfc), 'Sigc.shape=',np.shape(Sigc))
    if Gfc.ndim==1 and Sigc.ndim==1:# 1D case
        dim=1
    elif Gfc.ndim==4 and Sigc.ndim==4:# 4D case
        dim=4
    else:
        print('Error: Gfc and Sigc have wrong dimensions!',np.shape(Gfc),np.shape(Sigc))

    if EimpS.all() < 1000 :
        if dim==1:
            Gd = Gfc
            eimps = EimpS
            Sg = Sigc
            s_oo = Sigc[-1].real
            Sg0 = Sg - s_oo
            (A,C) = GetHighFrequency(Sg0,om)
            # at order 0 and 1, self-energy is just a constant. so this will give a error?
            # print( 'pert_energy_lib:A,C, for Sigma= ', A, C)
            # C=1.
            ff = Gd[:]*Sg0[:] - 1./(om*1j-eimps)*A/(om*1j-C)
            SDf= 2*sum(ff.real)/beta + ReturnHighFrequency(A,C,beta,eimps)
            # Note: I don't like this way. take const part of Sigma out, then add it back,and take analytical part of G, that should be good enough!
            # If take higher order of sigma there might be some issues.
        elif dim==4:

            Gd = Gfc
            eimps = EimpS
            Sg = Sigc
            s_oo = Sigc[-1,:,:,:].real
            Sg0 = Sg - s_oo[None,:,:,:]  # Sigma-s_oo
            # if sumabsSg0<0.1:
            #     print('Warning: sumabssg0 is almost 0')
            # plt.plot(Sg[:,0,0,0].real,label='Sg real')
            # plt.plot(Sg[:,0,0,0].imag,label='Sg imag')
            # plt.legend()
            # plt.grid()
            # plt.show()
            (A,C) = GetHighFrequency(Sg0,om[:,None,None,None])
            # print( 'A,C, for Sigma= ', A, C)
            # C=np.ones((knum,knum,knum))
            # A little note about this C. Here it is calculated first then set to 1. 
            # ideally we can choose any C and analytically this will work. So even if we set C=1, the result should be the same.
            # I guess this  is not so important for SigmaG, but for DeltaG, this C is important. But that is for energy but not here.
            ff = Gd*Sg0 - 1./(om[:,None,None,None]*1j-eimps[None,:,:,:])*A[None,:,:,:]/(om[:,None,None,None]*1j-C[None,:,:,:])
            # print('ff.shape=',np.shape(ff))
            SDf= (2*np.sum(ff.real)/beta + np.sum(ReturnHighFrequency(A,C,beta,eimps)))/knum**3
    return SDf

def fTrSigmaG_bf(om, Gfc, Sigc, EimpS, beta,knum=10):
    '''
    This is the brute force version of TrSigmaG. It is used to check the accuracy of the high frequency approximation.
    '''
    SDf=0.0
    # print('TrSigmaG: Gfc.shape=',np.shape(Gfc), 'Sigc.shape=',np.shape(Sigc))
    if Gfc.ndim==1 and Sigc.ndim==1:# 1D case
        dim=1
    elif Gfc.ndim==4 and Sigc.ndim==4:# 4D case
        dim=4
    else:
        print('Error: Gfc and Sigc have wrong dimensions!',np.shape(Gfc),np.shape(Sigc))

    if EimpS.all() < 1000 :
        if dim==1:
            Gd = Gfc
            eimps = EimpS
            Sg = Sigc
            s_oo = Sigc[-1].real
            Sg0 = Sg - s_oo
            #brute-force
            ff_bf=Gfc*Sigc
            SDf=(np.sum(ff_bf.real)*2)/beta
            # # a better way
            # ff = Gd[:]*Sg0[:] - 1./(om*1j-eimps)*s_oo
            # SDf= 2*sum(ff.real)/beta + ferm(eimps*beta)*s_oo

        elif dim==4:

            Gd = Gfc
            eimps = EimpS
            Sg = Sigc
            s_oo = Sigc[-1,:,:,:].real
            Sg0 = Sg - s_oo[None,:,:,:]  # Sigma-s_oo

            # ff = Gd*Sg0 - 1./(om[:,None,None,None]*1j-eimps[None,:,:,:])*s_oo[None,:,:,:]
            # SDf= (2*np.sum(ff.real)/beta + np.sum(ferm(eimps*beta)*s_oo))/knum**3
            ff_bf=Gd*Sg
            SDf=2*np.sum(ff_bf.real)/beta/knum**3
    return SDf

def fTrSigmaG1(om, Gfc, Sigc, EimpS, beta,knum=10):
    '''
    This is a better but simple version of TrSigmaG. It deducts the const part of Sigma, then add it back, and take the analytical part of G.
    '''
    SDf=0.0
    # print('TrSigmaG: Gfc.shape=',np.shape(Gfc), 'Sigc.shape=',np.shape(Sigc))
    if Gfc.ndim==1 and Sigc.ndim==1:# 1D case
        dim=1
    elif Gfc.ndim==4 and Sigc.ndim==4:# 4D case
        dim=4
    else:
        print('Error: Gfc and Sigc have wrong dimensions!',np.shape(Gfc),np.shape(Sigc))

    if EimpS.all() < 1000 :
        if dim==1:
            Gd = Gfc
            eimps = EimpS
            Sg = Sigc
            s_oo = Sigc[-1].real
            Sg0 = Sg - s_oo
            #brute-force
            ff_bf=Gfc*Sg0
            SDf=(np.sum(ff_bf.real)*2)/beta
            # # a better way
            # ff = Gd[:]*Sg0[:] - 1./(om*1j-eimps)*s_oo
            # SDf= 2*sum(ff.real)/beta + ferm(eimps*beta)*s_oo

        elif dim==4:

            Gd = Gfc
            eimps = EimpS
            Sg = Sigc
            s_oo = Sigc[-1,:,:,:].real
            Sg0 = Sg - s_oo[None,:,:,:]  # Sigma-s_oo

            # ff = Gd*Sg0 - 1./(om[:,None,None,None]*1j-eimps[None,:,:,:])*s_oo[None,:,:,:]
            # SDf= (2*np.sum(ff.real)/beta + np.sum(ferm(eimps*beta)*s_oo))/knum**3
            ff_bf=Gd*Sg0
            SDf=2*np.sum(ff_bf.real)/beta/knum**3
    return SDf

def LogG(omega,Gfc,EimpS,beta):# modify this for 4D version. But the trace limits that only local part of G counts.
    """
    Tr(log(-G_imp))
    Note: the inputs of GF should only has positive freq data. negative freq data  will be automatically covered.
    """
    # Here LogGimp is up to a i*pi factor with Log(-Gimp). Since G(iom)=G*(-iom) so TrLogG should be exactly real.
    def NonIntF0(beta,Ene):
        # if beta*Ene>200: return 0.0
        # if beta*Ene<-200: return Ene
        return -np.log(1+np.exp(-Ene*beta))/beta
    if Gfc.ndim==1:# 1D case
        dim=1
    elif Gfc.ndim==4:# 4D case
        dim=4
        knum=np.shape(Gfc)[1]
    else:
        print('Error: Gfc and Sigc have wrong dimensions!',np.shape(Gfc))
    lnGimp=0.

    # print('EimpS=', EimpS, ', real(iom-1/G)', (-1/Gfc[-1,:,:,:]).real)
    if dim==4:
        if EimpS.all() < 1000 :
            Gd = Gfc
            eimps = EimpS

            # CC = omega[:,None,None,None]*1j-eimps[None,:,:,:]-1/Gd # This looks like delta. but sometimes it is hard to get exact eimp.
            # A,C = GetHighFrequency(CC,omega)

            ff = np.log(-Gd)-np.log(-1./(omega[:,None,None,None]*1j-eimps[None,:,:,:]))# - 1./(omega[:,None,None,None]*1j-eimps[None,:,:,:])*A/(omega[:,None,None,None]*1j-C)
            lnGimp = np.sum(2*sum(ff.real)/beta+NonIntF0(beta,eimps))/knum**3#+ReturnHighFrequency(A,C,beta,eimps))
            # print('ff=', np.sum(2*sum(ff)/beta)/knum**3)
            # print('A,C=', A,C)
        else:
            sum_imp = 0.0
            Fnint_imp = 0.0
            lnGimp_i = 0.0
        # print('Tr(log(-Gimp))*deg=', lnGimp)
    elif dim==1:
        if EimpS < 1000 :
            Gd = Gfc
            eimps = EimpS
            CC = omega*1j-eimps-1/Gd
            A,C = GetHighFrequency(CC,omega)
            ff = np.log(-Gd)-np.log(-1./(omega*1j-eimps))# - 1./(omega*1j-eimps)*A/(omega*1j-C)
            lnGimp = 2*sum(ff.real)/beta+NonIntF0(beta,eimps)#+ReturnHighFrequency(A,C,beta,eimps)
            # print('A,C=', A,C)
            # plt.plot(-Gd.real,label='-Gd real')
            # plt.plot(-Gd.imag,label='-Gd imag')
            # plt.plot((-1./(omega*1j-eimps)).real,label='1/(iom-eimps) real')
            # plt.plot((-1./(omega*1j-eimps)).imag,label='1/(iom-eimps) imag')
            # plt.legend()
            # plt.grid()
            # plt.title('eimps='+str(eimps))
            # plt.show()

        else:
            sum_imp = 0.0
            Fnint_imp = 0.0
            lnGimp_i = 0.0
    return lnGimp

def LogGdiff(omega,Gimp,Gk,beta):
    " Tr(log(-G_imp)) - Tr(log(-G_k))=Tr(log(-G_imp/G_k)) since they are small so it might be accurate."
    knum=np.shape(Gk)[1]
    ff = np.log(Gk/Gimp[:,None,None,None])
    logGdiff = 2*np.sum(ff.real)/beta/knum**3
    return logGdiff

def LogG_EigVal(S11,S12,S22,beta,mu,knum=10,nfreq=500):

    # also, we need the eigenvalues of the G matrix, for the TrlogG term of the lattice system:
    # G+-=iom+mu-SigPM+-sqrt(SigAFM**2+(eps_k+Sig12))
    om= (2*np.arange(nfreq)+1)*np.pi/beta
    disper=lib.calc_disp(knum)
    SigPM=0.5*(S11+S22)
    SigAFM=0.5*(S11-S22)
    Siglambda11=SigPM-np.sqrt(SigAFM**2+(S12+disper[None,:,:,:])**2)#
    Siglambda22=SigPM+np.sqrt(SigAFM**2+(S12+disper[None,:,:,:])**2)#
    Glambda1=1/(om[:,None,None,None]*1j+mu-Siglambda11[nfreq:])
    Glambda2=1/(om[:,None,None,None]*1j+mu-Siglambda22[nfreq:])
    EimpS1=Siglambda11[-1,:,:,:].real-mu
    EimpS2=Siglambda22[-1,:,:,:].real-mu
    LogGeigval=LogG(om,Glambda1,EimpS1,beta)+LogG(om,Glambda2,EimpS2,beta)
    return LogGeigval.real

def H0G(G12,T,U):
    '''
    This function is designed to calculate energy for a given GF or self energy:
    E=Tr(H_0*G)+Tr(Sigma*G)/2
    '''
    knum=10
    beta=1/T
    mu=U/2
    disp=lib.calc_disp(knum)
    H_0G=-mu*1+np.sum(disp[None,:,:,:]*(G12*2)).real/knum**3/beta# here we should have a -n*mu but here at half-filling 
    return H_0G

def back_integration(Tlist,Elist,lastentropy):
    Slist=np.zeros_like(Tlist)
    Slist[-1]=lastentropy
    for iT,T in enumerate(Tlist[:-1]):
        Slist[iT]=lastentropy-Elist[-1]/Tlist[-1]+Elist[iT]/Tlist[iT]-np.sum(Elist[iT:-1]/Tlist[iT:-1]/Tlist[iT:-1]+Elist[iT+1:]/Tlist[iT+1:]/Tlist[iT+1:])*0.01/2# dT=0.01
    return Slist

def PertFreeEnergy(Sigma11,Sigma22,Sigma12,U,T,mode=0):
    '''
    order: does not matter. only comes in the filenames. at order 0 and 1, sigma is not k-dependent.
    Sigma: perturbation-corrected self-energy. usually in 4D array.
    U: Hubbard U
    T: temperature
    alpha: AFM splitting
    Here i am trying to split the calculation of total and free energies.
    '''
    debug=1
    beta=1./T
    knum=np.shape(Sigma11)[1]
    nfreq=int(np.shape(Sigma11)[0]/2)
    mu=U/2
    DMFTcheck=0# if this is 1 we use DMFT self energy to check, and also print the log file

    # before we start, a few things to explain:
    # to calculate the corrected energy we need energy for the impurity system.
    # however the result may vary when we have different orders and alphas.
    # currently, let's say, for any incoming self-energies, we try our best to calculate the lattice free energy, 
    # i.e. in free energy we include as many orders of nonlocal corrections in phi as we can.
    # Thus, the desired output should be like this:
    # File name: U_T.dat      (the order is the order of perturbation of GF, but not order of free energy correction!)
    # imp (Fimp) (Eimp)
    # (alpha) (F at order 0) (E at order 0) (F at order 1) (E at order 1) (F at order 2) (E at order 2) (F at order 3) (E at order 3)
    # ......
    
    # about the files:
    # first, cmpEimp (ctqmc or bold) will generate energy files about impurity. this is stored with DMFT files, like Sig, Gf, delta, ...
    # this first directory is called 'fileenergy'
    # and we should also have another file which is about the perturbed energy, which is stored in perturbation/energydata.
    # this second directory is called 'filepertenergy'
    if mode==0:
        dir='../files_boldc/{}_{}/'.format(U,T)# dir or DMFT related files. currently only boldc is supported.
    elif mode==1:
        dir='../files_ctqmc/{}_{}/'.format(U,T)
    fileenergy='../perturbation/impenergydata/{}_{}.txt'.format(U,T)
    filepertenergy='../perturbation/energydata/{}_{}.log'.format(U,T)# this is a log file which check if the code is correct.


    # if os.path.exists(fileenergy)==False:# energy file not generated yet 
    gen_energy_files(U,T,mode)# maybe better to renew the energy file every time.
    # with open(filepertenergy, 'w') as f:# first one use write mode to clean the file. then use append mode.
    # print('Perturbed Free Energy Calculation: U=',U,'T=',T)
    Phi_DMFT=find_last_value(fileenergy,'Phi_DMFT=')
    Fimp=find_last_value(fileenergy,'Fimp=')
    # for the 2 quantities below, i prefer not to read them but calculate them using my own functions. this will ensure they are calculated in the same way as in the main code to aviud systematic error.
    logGimp=find_last_value(fileenergy,'logGimp=')
    TrSigmaGimp=find_last_value(fileenergy,'TrSigmaG=')
    Eimp=find_last_value(fileenergy,'Eimp=')
    # return 0
    
    # Read DMFT G and Sigma from files. Things about DMFT and impurity solver.
    dirsig=dir+'Sig.out'
    dirg=dir+'Gf.out'
    SigimpA,SigimpB=readDMFT(dirsig,nfreq)
    SigimpB=U-SigimpA.real+1j*SigimpA.imag
    GimpAshort,GimpBshort=readDMFT(dirg,nfreq)
    GimpBshort=-GimpAshort.conjugate()
    GimpA=lib.ext_g(GimpAshort)
    GimpB=lib.ext_g(GimpBshort)
    if debug==2:
        # plt.plot(SigimpA.real,label='SigimpA real')
        # plt.plot(SigimpA.imag,label='SigimpA imag')
        # plt.plot(SigimpB.real,label='SigimpB real')
        # plt.plot(SigimpB.imag,label='SigimpB imag')
        plt.plot(GimpA.real,label='GimpA real')
        plt.plot(GimpA.imag,label='GimpA imag')
        plt.plot(GimpB.real,label='GimpB real')
        plt.plot(GimpB.imag,label='GimpB imag')
        plt.legend()
        plt.grid()
        plt.show()

    # generating Gimp with high quality
    SigDMFT11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    SigDMFT22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    SigDMFT12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    SigDMFT11+=lib.ext_sig(SigimpA)[:,None,None,None]
    SigDMFT22+=lib.ext_sig(SigimpB)[:,None,None,None]
    zDMFT_1=lib.z4D(beta,mu,SigDMFT11,knum,nfreq)
    zDMFT_2=lib.z4D(beta,mu,SigDMFT22,knum,nfreq)
    GDMFT11_iom,G12_iom=lib.G_iterative(knum,zDMFT_1,zDMFT_2,SigDMFT12)
    GDMFT22_iom=-GDMFT11_iom.conjugate()
    GDMFTloc11_iom=np.sum(GDMFT11_iom,axis=(1,2,3))/knum**3
    GDMFTloc22_iom=np.sum(GDMFT22_iom,axis=(1,2,3))/knum**3

    

    # Phi(n)[Gimp] is much easier, we can call the function to get local version of self-energy diagrams.
    simp11_oo=SigimpA[-1].real-mu # This should be something like delta, the half gap betn up and dn bands.
    Sigimp1_11,Sigimp1_22=libimp.pertimp_func(GDMFTloc11_iom,simp11_oo,beta,U,knum,order=1)
    Sigimp2_11,Sigimp2_22=libimp.pertimp_func(GDMFTloc11_iom,simp11_oo,beta,U,knum,order=2)
    Sigimp3_11,Sigimp3_22=libimp.pertimp_func(GDMFTloc11_iom,simp11_oo,beta,U,knum,order=3)
    # print('Sigma diagrams of impurity calculated')



    if DMFTcheck==1:
        #take DMFT sigma and G:
        Sig11=SigDMFT11
        Sig22=SigDMFT22
        Sig12=SigDMFT12
        # note: DMFT means order0, which is still for the system under B.
        # with open(filepertenergy, 'w') as f:
        # print('Using DMFT lattice GF and DMFT self-energy to calculate free energy. Input sigma neglected.')
    else:# take the best perturbative sigma
        Sig11=Sigma11
        Sig22=Sigma22
        Sig12=Sigma12
        # with open(filepertenergy, 'a') as f:
        # print('Using input GF and self-energy to calculate free energy')

        
    if debug==2:
        plt.plot(Sig11[nfreq:,0,0,0].real,label='Sigma11 real')
        plt.plot(Sig11[nfreq:,0,0,0].imag,label='Sigma11 imag')
        plt.legend()
        plt.grid()
        plt.show()




 

    # About perturbed GF, not DMFT:
    # Get best GF
    z_1=lib.z4D(beta,mu,Sig11,knum,nfreq)
    z_2=lib.z4D(beta,mu,Sig22,knum,nfreq)
    G11_iom,G12_iom=lib.G_iterative(knum,z_1,z_2,Sig12)
    G22_iom=-G11_iom.conjugate()
    # print('GF calculated')
    if debug==2:
        plt.plot(G11_iom[nfreq:,0,0,0].real,label='G11_iom real')
        plt.plot(G11_iom[nfreq:,0,0,0].imag,label='G11_iom imag')
        plt.legend()
        plt.grid()
        plt.show()


    # geneate accurate GF in imaginary time.
    s11_oo = Sig11[-1,:,:,:].real# currently this is a 3d array, each k point has a s_oo.
    EimpS11 = -mu+s11_oo # this is also a 3d array. G~1/(iom-eimp), so we need eimp.
    s22_oo = Sig22[-1,:,:,:].real
    EimpS22 = -mu+s22_oo
    # print('s_oo:',s11_oo[0,0,0],s22_oo[0,0,0])
    G11_tau=libfft.fermion_fft_diagG_4D(knum,G11_iom,beta,EimpS11)
    G12_tau=libfft.fast_ft_fermion(G12_iom,beta)
    G22_tau=libfft.fermion_fft_diagG_4D(knum,G22_iom,beta,EimpS22)
    # print('GF in tau calculated')
    if debug==2:
        plt.plot(G11_tau[:,0,0,0].real,label='G11_tau real')
        plt.plot(G11_tau[:,0,0,0].imag,label='G11_tau imag')
        plt.legend()
        plt.grid()
        plt.show()


    # Get corrections of Phi(n). We do not need the detailed sigma of each order. Essentially Phi can also be done as Tr(Sigma*G) but use specific sigma and G (and take care of symmetry factors!)
    # An important point is we have to construct Phi(n)[Gbest] from beginning since we don't have it before. so we have to call for the function to generate self-energy diagrams, then get Phi(n).

    #First, we have to calculate the corresponding self-energy diagrams of Gbest. for free energy.
    Sig1_11,Sig1_22=diagrams.sig1(G11_iom,G22_iom,knum,U,beta)
    Sig2_11,Sig2_12=diagrams.sig2(G11_tau,G12_tau,G22_tau,knum,nfreq,U,beta)
    # don't put counter terms here!
    Sig3_11,Sig3_12=diagrams.sig3(G11_iom,G12_iom,G11_tau,G12_tau,G22_tau,knum,nfreq,U,beta)# all diagrams of 3rd order should have the same symmetry factor.
    if debug==2:
        # plt.plot(Sigimp1_11.real,label='Sigimp1_11 real')
        # plt.plot(Sigimp1_11.imag,label='Sigimp1_11 imag')
        plt.plot(Sigimp2_11.real,label='Sigimp2_11 real')
        plt.plot(Sigimp2_11.imag,label='Sigimp2_11 imag')
        plt.plot(Sig2_11[:,0,0,0].real,label='Sig2_11 real')
        plt.plot(Sig2_11[:,0,0,0].imag,label='Sig2_11 imag')
        plt.legend()
        plt.grid()
        plt.show()

        # plt.plot(Sigimp3_11.real,label='Sigimp3_11 real')
        # plt.plot(Sigimp3_11.imag,label='Sigimp3_11 imag')
        # plt.plot(Sig3_11[:,0,0,0].real,label='Sig3_11 real')
        # plt.plot(Sig3_11[:,0,0,0].imag,label='Sig3_11 imag')
        # plt.legend()
        # plt.grid()
        # plt.show()





    #Get Perturbed Free energy
    om= (2*np.arange(nfreq)+1)*np.pi/beta
    # Note: here we used some symmetry of the half-filled system. Here 11 and 22 actually means up and dn spin. 
    # And since we have G11(iom)=-G22(iom).conj, we have phi11*11=phi22*22.conj. but since phi is real, we have phi11=phi22. 
    # What we are calculating is the phi of a double unit cell.so we need 11*11, 12*21, 22*22, 21*12, which is twice of 11*11 and 12*21.
    # It is a bad idea to simply do a freq summation, and even use the simple trick as TrsigmaG will not work.
    # the reason is G11 here scales like 1/(iom-eimp1)+1/(iom-eimp2), thus only one Eimp will not work for this.
    # An alternative way to do the first order is U*n1*n2 and this simple approximation will give decent Phi(1), also roughly corresponds to the result of Phi from bold or ctqmc.
    Phi_1=Sig1_11*Sig1_22/2*2/U# updn and dnup. sym factor 2.
    Phiimp_1=Sigimp1_11*Sigimp1_22/2*2/U
    Phi_2=fTrSigmaG_bf(om, G11_iom[nfreq:,:,:,:], Sig2_11[nfreq:,:,:,:]/4, EimpS11, beta,knum)*2
    Phi_3=fTrSigmaG_bf(om, G11_iom[nfreq:,:,:,:], Sig3_11[nfreq:,:,:,:]/6, EimpS11, beta,knum)*2
    Phi_2_offdiag=fTrSigmaG_bf(om, G12_iom[nfreq:,:,:,:], Sig2_12[nfreq:,:,:,:]/4, np.zeros((nfreq,knum,knum,knum)), beta,knum)*2
    Phi_3_offdiag=fTrSigmaG_bf(om, G12_iom[nfreq:,:,:,:], Sig3_12[nfreq:,:,:,:]/6, np.zeros((nfreq,knum,knum,knum)), beta,knum)*2
    Phiimp_2=fTrSigmaG_bf(om, GDMFTloc11_iom[nfreq:], Sigimp2_11[nfreq:]/4,  -mu+SigimpA[-1], beta,knum)*2
    Phiimp_3=fTrSigmaG_bf(om, GDMFTloc11_iom[nfreq:], Sigimp3_11[nfreq:]/6,  -mu+SigimpA[-1], beta,knum)*2
    Phi=Phi_DMFT+Phi_1-Phiimp_1+Phi_2+Phi_2_offdiag-Phiimp_2+Phi_3+Phi_3_offdiag-Phiimp_3
    # print('Phi calculated')
    if DMFTcheck==1:
        with open(filepertenergy, 'a') as f:
            print('Phidis_1=',Phi_1,'\tPhidis_2=',Phi_2,'\tPhidis_3=',Phi_3,file=f)#
            print('Phiimp_1=',Phiimp_1,'\tPhiimp_2=',Phiimp_2,'\tPhiimp_3=',Phiimp_3,file=f)#
            print('Phi_2_offdiag=',Phi_2_offdiag,'\tPhi_3_offdiag=',Phi_3_offdiag,file=f)#
            print('Phi_DMFT=',Phi_DMFT,'Phi_corrected=',Phi,file=f)
    
    # finally, calculate F, and compare with Fimp. 
    # Note: to compare the result from DMFT and DMFT+perturbation, we have to use the same way to calculate the corresponding terms to avoid any systematic error.
    # Now the myTrSigmaGimp and MyLogGimp are exactly the same as the ones from boldc_ctqmc.log from the impurity solver.
    n11=(np.sum(G11_iom).real/knum**3/beta+1/2)
    n22=(np.sum(G22_iom).real/knum**3/beta+1/2)
    n11imp=(np.sum(GDMFTloc11_iom).real/beta+1/2)
    n22imp=(np.sum(GDMFTloc22_iom).real/beta+1/2)

    TrSigmaG=fTrSigmaG(om, G11_iom[nfreq:], Sig11[nfreq:], EimpS11, beta,knum)+fTrSigmaG(om, G22_iom[nfreq:], Sig22[nfreq:], EimpS22, beta,knum)+fTrSigmaG_bf(om, G12_iom[nfreq:], Sig12[nfreq:], np.zeros((nfreq,knum,knum,knum)), beta,knum)*2
    TrSigmaG+=np.sum(n11*s11_oo+n22*s22_oo)/knum**3 # remember to add the infinite part!
    # print('TrSigmaG calculated')
    # myTrSigmaGimp=fTrSigmaG(om, GimpA[nfreq:], SigimpA, -mu+SigimpA[-1].real-B, beta,knum)+fTrSigmaG(om, GimpB[nfreq:], SigimpB, -mu+SigimpB[-1].real+B, beta,knum)
    myTrSigmaGimp=fTrSigmaG(om, GDMFTloc11_iom[nfreq:], SigimpA, -mu+SigimpA[-1].real, beta,knum)+fTrSigmaG(om, GDMFTloc22_iom[nfreq:], SigimpB, -mu+SigimpB[-1].real, beta,knum)
    myTrSigmaGimp+=np.sum(n11imp*SigimpA[-1].real+n22imp*SigimpB[-1].real)
    # print('myTrSigmaGimp calculated')

    # TrlogG_check=LogG(om,G11_iom[nfreq:],EimpS11,beta)+LogG(om,G22_iom[nfreq:],EimpS22,beta)
    myLogGimp=LogG(om,GDMFTloc11_iom[nfreq:],-mu+SigimpA[-1].real,beta)+LogG(om,GDMFTloc22_iom[nfreq:],-mu+SigimpB[-1].real,beta)

    TrlogG=LogG_EigVal(Sig11,Sig12,Sig22,beta,mu)
    H0_G=H0G(G12_iom,T,U)
    Edisp=H0_G+TrSigmaG/2
    # about the Free energy: for off-diagonal F we should take TrLogG=log(DetG), or diagonalize G to get eigenvals.
    Fdisp=TrlogG-TrSigmaG+Phi

    # TrlogG_alter=myLogGimp+logG_diff
    if DMFTcheck==1:
        with open(filepertenergy, 'a') as f:
            print('n11=',n11,'n22=',n22, file=f)
            print('MyIMP(check) TrlogGimp={:.6f}'.format(myLogGimp),'TrSigmaGimp={:.6f}'.format(myTrSigmaGimp),'Phiimp={:.6f}'.format(Phi_DMFT),
                  'Fimp={:.6f}'.format(myLogGimp-myTrSigmaGimp+Phi_DMFT), file=f) 
            print('IMPSOLVER    TrlogG={:.6f}'.format(logGimp),'TrSigmaG={:.6f}'.format(TrSigmaGimp),'Phi={:.6f}'.format(Phi_DMFT),'F={:.6f}'.format(Fimp), file=f) 
            # print('DMFT         TrlogG={:.6f}'.format(logGimp),'TrSigmaG={:.6f}'.format(TrSigmaGimp),'Phi={:.6f}'.format(Phi_DMFT),'F={:.6f}'.format(Fimp), file=f) 
            print('DMFT+PERT    TrlogG={:.6f}'.format(TrlogG),'TrSigmaG={:.6f}'.format(TrSigmaG),'Phi={:.6f}'.format(Phi),'F={:.6f}'.format(Fdisp), file=f)

        # Essentially we could calculate various F, from the worst to the best:
        # 1. Fimp[Gimp]: directly from impurity solver. (IMPSOLVER)
        # 2. Fimp[G_DMFT]: use dispersive G_DMFT to calculate F, which improves the first term. But for Phi still only local diagrams.
        # 3. F[G_DMFT]: use dispersive G_DMFT to calculate F, and also add nonlocal correction to Phi. (DMFT+PERT at Gbest=0)
        # 4. F[Gbest]: use the perturbative corrected GF (should be better than G_DMFT), and also add nonlocal correction to Phi. (DMFT+PERT at Gbest=1)
    # give a warning if 2 impurity quantities are not close enough.
    if abs(myLogGimp-logGimp)>0.001 or abs(myTrSigmaGimp-TrSigmaGimp)>0.001:
        print('Warning: the calculated impurity quantities are not close enough to the ones from impurity solver!',myLogGimp,logGimp,myTrSigmaGimp,TrSigmaGimp)
    # print(Fimp, Eimp,Fdisp,Edisp)
    # print('logGimp=',logGimp,'logGdisp=',TrlogG,'TrSigmaGimp=',TrSigmaG)
    # print('Phiimp=',Phi_DMFT,'Phiimp3orders=',Phiimp_1+Phiimp_2+Phiimp_3,'Phidisp3orders=',Phi_1+Phi_2+Phi_3)
    print('Edisp={:.6f},H0G={:.6f},TrSigmaG/2={:.6f},TrSigmaG_inf={:.6f}'.format(Edisp,H0_G,TrSigmaG/2,np.sum(n11*s11_oo+n22*s22_oo)/knum**3))
    print('Fdisp={:.6f},TrlogG={:.6f},TrSigmaG={:.6f},Phiimp={:.6f}, Phidisp={:.6f}\n'.format(Fdisp,TrlogG,TrSigmaG,Phi_DMFT,Phi))
    return Fimp, Eimp,Fdisp,Edisp
    # return F, TrlogG, TrSigmaG, Phi, F_alter, TrlogG_alter






def imp_phase_transition():
    # U=4.
    # Tlist=np.array([0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24]) #0.11,0.12,0.13,
    # Tlist=np.array([0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69])#do boldc of 0.41    
    # U=5.
    # Tlist=np.array([0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31])
    # U=6.
    # Tlist=np.array([0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39])#0.3,
    U=8.
    Tlist=np.array([0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,
                    0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58])#do boldc of 0.410.15,
    # U=10.
    # Tlist=np.array([0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,
    #                 0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6,0.61,0.62])#
    # Tlist2=np.array([0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,
    #                 0.49,0.5,0.51,0.52,0.53])#,0.54,0.55,0.56,0.57,0.58,0.59

    # U=12.
    # Tlist=np.array([0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,
    #                 0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69])#

    Sig11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    Sig22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    Sig12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)

    #boldc
    Fimp_arr=np.zeros_like(Tlist)
    Eimp_arr=np.zeros_like(Tlist)
    Fdisp_arr=np.zeros_like(Tlist)
    Edisp_arr=np.zeros_like(Tlist)

    for iT, T in enumerate(Tlist):
        Fimp_arr[iT],Eimp_arr[iT],Fdisp_arr[iT],Edisp_arr[iT]=PertFreeEnergy(0,Sig11,Sig22,Sig12,U,T,0)
    entropy=(Eimp_arr-Fimp_arr)/Tlist
    entropydisp=(Edisp_arr-Fdisp_arr)/Tlist
    lastent=entropy[-1]
    lastentdisp=entropydisp[-1]
    entropy_int=back_integration(Tlist,Eimp_arr,lastent)
    entropydisp_int=back_integration(Tlist,Edisp_arr,lastentdisp)

    #ctqmc
    # Fimp_arr1=np.zeros_like(Tlist)
    # Eimp_arr1=np.zeros_like(Tlist)
    # Fdisp_arr1=np.zeros_like(Tlist)
    # Edisp_arr1=np.zeros_like(Tlist)
    # for iT, T in enumerate(Tlist):
    #     Fimp_arr1[iT],Eimp_arr1[iT],Fdisp_arr1[iT],Edisp_arr1[iT]=PertFreeEnergy(0,Sig11,Sig22,Sig12,U,T,1)
    # entropy1=(Eimp_arr1-Fimp_arr1)/Tlist
    # lastent1=entropy1[-1]
    # entropy_int1=back_integration(Tlist,Eimp_arr1,lastent1)



    # plt.plot(Tlist,Fimp_arr,label='Fimp boldc')
    # plt.plot(Tlist,Eimp_arr,label='Eimp boldc')
    plt.plot(Tlist,Fdisp_arr,label='F latt boldc')
    plt.plot(Tlist,Edisp_arr,label='E latt boldc')   
    # plt.plot(Tlist,Fimp_arr1,label='Fimp ctqmc')
    # plt.plot(Tlist,Eimp_arr1,label='Eimp ctqmc')    
    plt.xlabel('T')
    plt.ylabel('Energy')
    plt.title('Thermodynamics of DMFT: U={}'.format(U))
    plt.legend()
    plt.show()


    # plt.plot(Tlist,entropy,label='(E-F)/T imp boldc')
    # plt.plot((Tlist[1:]+Tlist[:-1])/2,-(Fimp_arr[1:]-Fimp_arr[:-1])/0.01,label='-dF/dT imp boldc')
    # plt.plot(Tlist,entropy_int,label='thermo imp boldc')

    plt.plot(Tlist,entropydisp,label='(E-F)/T latt boldc')
    plt.plot((Tlist[1:]+Tlist[:-1])/2,-(Fdisp_arr[1:]-Fdisp_arr[:-1])/0.01,label='-dF/dT latt boldc')
    plt.plot(Tlist,entropydisp_int,label='thermo latt boldc')


    # # plt.plot(Tlist,entropy1,label='(E-F)/T ctqmc')
    # # plt.plot((Tlist[1:]+Tlist[:-1])/2,-(Fimp_arr1[1:]-Fimp_arr1[:-1])/0.01,label='-dF/dT ctqmc')   
    # # plt.plot(Tlist,entropy_int1,label='back_int ctqmc')
    # # # plt.plot(Tlist,(Edisp_arr-Fdisp_arr)/Tlist,label='disp1')    
    plt.legend()
    plt.xlabel('T')
    plt.ylabel('Entropy')
    plt.title('Entropy of lattice: U={}'.format(U))
    plt.show()


    plt.plot((Tlist[1:]+Tlist[:-1])/2,(entropydisp[1:]-entropydisp[:-1])/0.01,label='dS/dT latt boldc')      
    plt.legend()
    plt.xlabel('T')
    plt.ylabel('dS/dT')
    plt.title('dS/dT: U={}'.format(U))
    plt.show()
    return 0

if __name__ == "__main__":
    # just for testing.
    # if directly run this file, this will do the perturbation without variational part. 
    nfreq=500
    knum=10
    U=8.0
    T=0.25
    if len(sys.argv)>=3:
        U=float(sys.argv[1])
        T=float(sys.argv[2])
    imp_phase_transition()
    # Sig11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    # Sig22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    # Sig12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    # F=PertFreeEnergy(0,Sig11,Sig22,Sig12,U,T,0.0,0)