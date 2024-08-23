#!/usr/bin/env python
# Author: Kristjan Haule, March 2007-2017
from scipy import *
from scipy import interpolate
from numpy import *
import json
import copy
import os,sys,subprocess
sys.path.append('../python_src/')
import hilbert
from perturb_lib import *
import matplotlib.pylab as plt
#from pylab import *

"""
This module runs ctqmc impurity solver for one-band model.
The executable should exist in directory params['exe']
"""
def cleanfile():
    cmd='rm '+fileS+' '+fileG+' '+fileD+' '+filelog+' debug* Sigma* status* Sig* PPG* Gf* Delta* PPSigma* dF* dG* dSig* histogram* uls.dat sampled_data f00* g00* sampled_data* Stval* uls*'  
    # subprocess.call('rm Gf.out PARAMS PPSigma.OCA Sig.out debu* dF* dG* diags* dSig* gf* histogram* PPG* PPSigma* status* *.OCA ctqmc.log Sig.out_Dyson uls.dat Delta.inp sampled_data', shell=True)
    subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr) 
    return 0


def Delta_DMFT_var(sig1,sig2,U,T,knum=10,a=1):
    mu=U/2
    beta=1/T
    n=sig1.size
    sigA=sig1
    sigB=U-sig1.real+1j*sig1.imag    
    # if sig1[-1].real>sig2[-1].real:
    #     sigA=sig1
    #     sigB=U-sig1.real+1j*sig1.imag
    # else:
    #     sigA=sig2
    #     sigB=U-sig2.real+1j*sig2.imag
    om= (2*np.arange(n)+1)*np.pi/beta
    iom=1j*om
    Sigma11=np.zeros((2*n,knum,knum,knum),dtype=complex)
    Sigma11+=ext_sig(sigA)[:,None,None,None]
    Sigma22=np.zeros((2*n,knum,knum,knum),dtype=complex)
    Sigma22+=ext_sig(sigB)[:,None,None,None]
    z_1=z4D(beta,mu,Sigma11,knum,n) #za=iom+mu-SigA+B
    z_2=z4D(beta,mu,Sigma22,knum,n) #zb=iom+mu-SigB-B
    G11_iom=G_11(knum,z_1,z_2)
    G11loc_iom=np.sum(G11_iom,axis=(1,2,3))[n:]/knum**3
    G22loc_iom=-G11loc_iom.conjugate()
    Gloc_inv_11=1/G11loc_iom
    Gloc_inv_22=1/G22loc_iom
    Delta_11=iom+mu-sigA-Gloc_inv_11
    Delta_22=iom+mu-sigB-Gloc_inv_22
    return Delta_11,Delta_22,G11loc_iom,G22loc_iom

def CreateInputFile(params):
    " Creates input file (PARAMS) for bolc-ctqmc solver"
    f = open('PARAMS', 'w')
    print(json.dumps(params,indent=4,separators=(',', ': ')), file=f)
    f.close()

def DMFT_SCC(fDelta,W=None):
    """This subroutine creates Delta.inp from Gf.out for DMFT on bethe lattice: Delta=t^2*G
    If Gf.out does not exist, it creates Gf.out which corresponds to the non-interacting model
    In the latter case also creates the inpurity cix file, which contains information about
    the atomic states.
    """
    if (os.path.exists(fileS)): # If output file exists, start from previous iteration
        # Gf = io.read_array(fileGf, columns=(0,-1), lines=(1,-1))
        # In the new Python, io.readarray is dropped and we should use loadtxt instead!
        Sf = loadtxt(fileS).T
        Sg_A = Sf[1,:]+Sf[2,:]*1j
        Sg_B = Sf[3,:]+Sf[4,:]*1j      

        om = Sf[0]
    else: # otherwise start from non-interacting limit
        print('Starting from non-interacting model')
        om = (2*arange(500)+1)*pi/params['beta']
        Sg_A=(Uc/2.+splitting+0j)*np.ones(500)
        Sg_B=(Uc/2.-splitting+0j)*np.ones(500)

    
    if (abs(om[0]-pi/params['beta'])>1e-6):
        # print('It seems input '+fileS+' correspond to different temperature, hence interpolating')
        raise ValueError('It seems input '+fileS+' correspond to different temperature')

    # Dlt_A,Dlt_B,GlocA,GlocB = hilbert.SCC_AFM(W, om, params['beta'], params['mu'], params['U'], Sg_A, Sg_B, False)
    Dlt_A,Dlt_B,GlocA,GlocB =Delta_DMFT_var(Sg_A,Sg_B,Uc,T,30)
    # plt.plot(HT_Dlt_A.real,label='HT_Dlt_A real')
    # plt.plot(HT_Dlt_A.imag,label='HT_Dlt_A imag')
    # plt.plot(HT_Dlt_B.real,label='HT_Dlt_B real')
    # plt.plot(HT_Dlt_B.real,label='HT_Dlt_B imag')
    # plt.plot(Dlt_A.real,label='Dlt_A real')
    # plt.plot(Dlt_A.imag,label='Dlt_A imag')
    # plt.plot(Dlt_B.real,label='Dlt_B real')
    # plt.plot(Dlt_B.real,label='Dlt_B imag')
    # plt.legend()
    # plt.grid()
    # plt.show()
    # Preparing input file Delta.inp. Since we shift this delta we have to do 2 times of impurity solver, seperately
    # f = open(fDelta, 'w')
    # for i,iom in enumerate(om):
    #     print(iom, Dlt_A[i].real, Dlt_A[i].imag, Dlt_B[i].real, Dlt_B[i].imag, file=f) 
    # f.close()
    f = open(fileGloc, 'w')
    for i,iom in enumerate(om):
        print(iom, GlocA[i].real, GlocA[i].imag, GlocB[i].real, GlocB[i].imag, file=f) 
    f.close()
    return Dlt_A,Dlt_B

    

def Diff(fg1, fg2):
    data1 = loadtxt(fg1).T
    data2 = loadtxt(fg2).T
    n_points = where(data1[0,:]>10.0)[0][0]
    diff = sum(abs(data1[:,:n_points]-data2[:,:n_points]))/(n_points*shape(data1)[0])
    return diff

if __name__ == "__main__":
    #Names of files
    fileS = 'Sig.out'
    fileG = 'Gf.out' 
    # fileS = 'Sig.OCA'
    # fileG = 'Gf.OCA'# impurity GF from impurity solver
    fileGloc = 'Gfloc.OCA'# local part of lattice GF
    fileD='Delta.inp'
    filelog='ctqmc.log'#bold_


    splitting=0.5
    Niter = 200
    Ms = 1e7
    mixing=0.3


    if (len(sys.argv)>=3):
        Uc=float(sys.argv[1])
        T=float(sys.argv[2])
        print('Uc=',Uc,'T=',T)
        dir='../files_boldc/{}_{}/'.format(Uc,T)

    if os.path.exists(dir):
        print('already have this directory: ', dir)
        # CAREFULL! delete any previous files
        subprocess.call('rm '+dir+'*', shell=True) 
    else:
        print('directory does not exist... make a new one')
        cmd_newfolder='mkdir '+dir
        subprocess.call(cmd_newfolder, shell=True)

    cleanfile()

    params={
    "exe"       : "mpirun -np 8 ../cache_boldc/boldc", # Path to executable
    "dos"       : "../python_src/DOS_3D.dat",     # non-interacting DOS
    "U"         : Uc,               # Coulomb U
    "mu"        : Uc/2.,            # chemical potential
    "B"         : 0,              # external magnetic field
    "beta"      : 1/T,               # inverse temperature
    "Norder"    : 3,                # the maximum perturbation order
  "N_min_order" : 2,                # the minimal order at which we run MC (the rest analytic)
    "Ms"        : Ms,               # Number of Monte Carlo steps at each iteration
    "Nbath"     :  2,               # paramagnetic               # seems we have to choose nbath=1?
    "mix"       : 0.5,              # mixing of pseudo self-energy
    "Nitt"      : 100,              # Number of iterations of the loop
    "cmpPhysG"  : False,            # Should we compute G00 at each step?
    "iseed"     :  0,               # seed for random number generator
    "V0norm"    : 0.001,            # constant to reweight higher order diagrams
    "tsample"   : 5,                # How often to make a measurement
    "svd_lmax"  : 20,               # using SVD functions up to cutoff
    "Delta"     : fileD,            # Input bath function hybridization
    "Nt"        : 300,              # How many times we try to change time before we consider change of diagram
    "Nd"        : 1,                # How many times we try to change diagram before we consider next change of time
    "converge_OCA": True,           # If histogram is not present, we coverge OCA before we start MC
    "diff_stop" : 0.01,             # condition to stop PP-loop
    "rescale"   : 0.4,              # during PP-loop we rescale last step values. But how much?
    "fastFilesystem": True,         # print extra file information
    "d_fraction": 3                 # We keep (Norder/d_fraction) propagators equal when proposing different diagram.
    }



    # Number of DMFT iterations


    # non-interacting DOS and its Hilbert transform (one way to do things)
    # x, Di = loadtxt(params['dos']).T
    # W = hilbert.Hilb(x,Di)
    stopcount=0
    itstop=999 # if we have not converged, we will stop after 999 iterations
    om=(2*arange(500)+1)*pi/params['beta']
    filediff='diff.dat'

    f = open(filediff, 'w')
    print(0, 0, 0, file=f)# creating a new file
    f.close()
    for it in range(1,Niter+1):
        # Constructing bath Delta.inp from Green's function
        deltaA,deltaB=DMFT_SCC(fileD)# this will generate delta A and B
        if it==1:
            prvdeltaA=deltaA
            prvdeltaB=deltaB
        else:
            deltaA=mixing*deltaA+(1-mixing)*prvdeltaA
            deltaB=mixing*deltaB+(1-mixing)*prvdeltaB
            prvdeltaA=deltaA
            prvdeltaB=deltaB
        f = open(fileD, 'w')
        for i,iom in enumerate(om):
            print(iom, deltaA[i].real, deltaA[i].imag, deltaB[i].real, deltaB[i].imag, file=f) 
        f.close()        
        cmd = 'cp '+fileD+' '+dir+fileD+'.'+str(it)# copying delta
        subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr)  
        cmd = 'cp '+fileGloc+' '+dir+fileGloc+'.'+str(it)# copy local part of lattice GF
        subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr) 

        CreateInputFile(params)
        # Running ctqmc
        print('Running ---- impsolver itt.: ', it, '-----')

        subprocess.call(params['exe'], shell=True,stdout=sys.stdout,stderr=sys.stderr)


        cmd = 'cp '+fileG+' '+dir+fileG+'.'+str(it)# copying impurity GF
        subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr)  
        cmd = 'cp '+fileS+' '+dir+fileS+'.'+str(it)
        subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr) # copying Sig




        if it>1:
            diff = Diff(fileG, fileGloc)
            diff1 = Diff(fileG, dir+fileG+'.'+str(it-1))
            print('Gimp-Gloc=', diff,'\tGimp-prevGimp=',diff1)
            #if (diff<3e-4 and params["Ms"]==Ms):
            #    params["Ms"] *= 3
            #    CreateInputFile(params)
            f = open(filediff, 'a')
            print(it, diff, diff1, file=f)
            f.close()

            if (diff1<7e-5) and stopcount==0: 
                itstop=it+15 # to be safe, we wait for 10 iterations
                stopcount=1
            if stopcount==1:
                print('converged, waiting for 15 iterations. will stop after',itstop)
            if it>itstop and it>25:# at least 40 iterations
                print('converged')
                break

    cmd = 'cp '+fileG+' '+dir+fileG# copying impurity GF
    subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr)  
    cmd = 'cp '+fileS+' '+dir+fileS
    subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr) # copying Sig
    cmd = 'cp '+fileD+' '+dir+fileD# copying delta
    subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr)  
    cmd = 'cp '+filelog+' '+dir+filelog# copying log
    subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr)  
    cmd = 'cp '+filediff+' '+dir+filediff# copying diff
    subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr)