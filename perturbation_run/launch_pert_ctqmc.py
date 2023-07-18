#!/usr/bin/env python
# Author: Kristjan Haule, March 2007-2017
from scipy import * 
import os,sys,subprocess
import numpy as np
import method_afm # this is method.py in the same directory 
import matplotlib.pyplot as plt 
from scipy import integrate
import hilbert
import perturb
"""
This module runs ctqmc impurity solver for one-band model.
The executable should exist in directory params['exe']
"""
# filedos='cubic_dos.dat'
filedos='DOS_3D.dat'
Uc=8
beta=20

if (len(sys.argv)==3):
    Uc=float(sys.argv[1])
    T=float(sys.argv[2])
    print('T=',T)
    print('Uc=',Uc)


dir='./files/{}_{}/'.format(Uc,T)

if os.path.exists(dir):
    print('already have this directory: ', dir)
    # CAREFULL! delete any previous files
    subprocess.call('rm '+dir+'*', shell=True) 
else:
    print('directory does not exist... make a new one')
    cmd_newfolder='mkdir '+dir
    subprocess.call(cmd_newfolder, shell=True)
    
subprocess.call('rm ctqmc.log Delta.inp Deltat.inp Gf.out PARAMS PPSigma.OCA Sig.out Delta.tau.00.000 Delta.tau.01.000 rDelta.tau.01.000 rDelta.tau.00.000', shell=True) 
for i in np.arange(8):
    subprocess.call('rm status.00{}'.format(i), shell=True)
   

params = {"exe":   ["mpirun ./ctqmc",          "# Path to executable"],
          "U":     [Uc,                 "# Coulomb repulsion (F0)"],
          "mu":    [Uc/2.,              "# Chemical potential"],
          "beta":  [1/T,                "# Inverse temperature"],
          "M" :    [1e7,                "# Number of Monte Carlo steps"],
          "mode":  ["SH",               "# S stands for self-energy sampling, M stands for high frequency moment tail"],
          "cix":   ["one_band.imp",     "# Input file with atomic state"],
          "Delta": ["Delta.inp",        "# Input bath function hybridization"],
          "tsample":[200,               "# how often to record the measurements" ],
          "nom":   [80,                 "# number of Matsubara frequency points to sample"],
        "svd_lmax":[30,                 "# number of SVD functions to project the solution"],
          "aom":   [1,                  "# number of frequency points to determin high frequency tail"],
          "GlobalFlip":[1000000,         "# how often to perform global flip"],
          "fastFilesystem":[1,          "#to produce Delta.tau"],
          }
icix="""# Cix file for cluster DMFT with CTQMC
# cluster_size, number of states, number of baths, maximum matrix size
1 4 2 1
# baths, dimension, symmetry, global flip
0       1 0 0
1       1 1 0
# cluster energies for unique baths, eps[k]
0 0
#   N   K   Sz size F^{+,dn}, F^{+,up}, Ea  S
1   0   0    0   1   2         3        0   0
2   1   0 -0.5   1   0         4        0   0.5
3   1   0  0.5   1   4         0        0   0.5
4   2   0    0   1   0         0        0   0
# matrix elements
1  2  1  1    1    # start-state,end-state, dim1, dim2, <2|F^{+,dn}|1>
1  3  1  1    1    # start-state,end-state, dim1, dim2, <3|F^{+,up}|1>
2  0  0  0
2  4  1  1   -1    # start-state,end-state, dim1, dim2, <4|F^{+,up}|2>
3  4  1  1    1
3  0  0  0
4  0  0  0
4  0  0  0
HB2                # Hubbard-I is used to determine high-frequency
# UCoulomb : (m1,s1) (m2,s2) (m3,s2) (m4,s1)  Uc[m1,m2,m3,m4]
0 0 0 0 0.0
# number of operators needed
0
"""


def CreateInputFile(params):
    " Creates input file (PARAMS) for CT-QMC solver"
    f = open('PARAMS', 'w')
    print('# Input file for continuous time quantum Monte Carlo', file=f)
    for p in params:
        print(p, params[p][0], '\t', params[p][1], file=f)
    f.close()

def DMFT_SCC(fDelta):
    """This subroutine creates Delta.inp from Gf.out for DMFT on bethe lattice: Delta=t^2*G
    If Gf.out does not exist, it creates Gf.out which corresponds to the non-interacting model
    In the latter case also creates the inpurity cix file, which contains information about
    the atomic states.
    """
    fileGf = 'Gf.out'
    filesig='../trial_sigma/{}_{}.dat'.format(Uc,T)
    #get sigma
    if (os.path.exists(filesig)): 
        sigma = np.loadtxt(filesig)
    else:# generate a trial sigma
        print("preparing trial sigma...")
        sigma=np.zeros((500,5),dtype=complex)
        for i in np.arange(500):
            omega=(2*i+1)*np.pi/params['beta'][0]
            sigma[i,0]=omega #omega
            sigma[i,1]=params['mu'][0]+0.01# Gaa,up,real
            sigma[i,2]=0# Gaa, up, imag
            sigma[i,3]=params['mu'][0]-0.01# Gaa, dn, real
            sigma[i,4]=0# Gaa, dn, imag
        print('writing trial sigma file')
        f = open('trial_sigma.dat', 'w')
        for i in range(np.shape(sigma)[0]):# consider the case that we may have many columns of Gf
            # print(Gf[i,0], 0.25*Gf[i,1], 0.25*Gf[i,2], file=f) # This is DMFT SCC: Delta = t**2*G (with t=1/2)
            # print(np.shape(Deltafile))
            for k in np.arange(np.shape(sigma)[1]):
                print(sigma[i,k],end='\t', file=f)# print in the same line
            print('', file=f)# switch to another line
        f.close()
        print("trial sigma finished",np.shape(sigma))

        f = open(params['cix'][0], 'w')
        print(icix, file=f)
        f.close()
    print('Getting Non-interacting DOS')


    # Preparing input file Delta.inp
    print('Generating and writing Delta...')
    # f = open(fDelta, 'w')
    # Delta=calc_Delta_afm(sigma,elist,dos,params['mu'][0])
    # for i in range(np.shape(sigma)[0]):# consider the case that we may have many columns of Gf
    #     for k in np.arange(5):
    #         print(Delta[i,k],end='\t', file=f)# print in the same line
    #     print('', file=f)# switch to another line
    # f.close()
    om = sigma[:,0].real
    Sg_A = sigma[:,1]+sigma[:,2]*1j
    Sg_B = sigma[:,3]+sigma[:,4]*1j
    # print(type(Sg_A),type(om),type(params['mu'][0]))
    # Dlt_A,Dlt_B = hilbert.SCC_AFM(W, om, params['beta'][0], params['mu'][0], params['U'][0], Sg_A, Sg_B, False)
    Dlt_A,Dlt_B=perturb.impurity_test(Sg_A,Sg_B,Uc,T,10)
    # Preparing input file Delta.inp
    f = open(fDelta, 'w')
    for i,iom in enumerate(om):
        print(iom, Dlt_A[i].real, Dlt_A[i].imag, Dlt_B[i].real, Dlt_B[i].imag, file=f) 
    f.close()
    cmd = 'cp Delta.inp '+dir+'Delta.inp.'+str(it)
    subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr)  # copying Gf
    print('Delta file is done!')


def Diff(fg1, fg2):
    data1 = np.loadtxt(fg1).transpose()
    data2 = np.loadtxt(fg2).transpose()
    diff = np.sum(abs(data1-data2))/(np.shape(data1)[0]*np.shape(data1)[1])
    return diff

# Number of DMFT iterations
Niter = 1

# Creating parameters file PARAMS for qmc execution
CreateInputFile(params)

#get dos
x, Di = np.loadtxt(filedos).T
W = hilbert.Hilb(x,Di)
print('Non-interacting DOS finished')

cmd_params = 'cp '+'PARAMS'+' '+dir+'PARAMS'
subprocess.call(cmd_params, shell=True)
cmd_cix = 'cp '+params['cix'][0]+' '+dir+params['cix'][0]
subprocess.call(cmd_cix, shell=True)
cmd_dos = 'cp '+filedos+' '+dir+filedos
subprocess.call(cmd_dos, shell=True)

for it in range(Niter):
    # Constructing bath Delta.inp from Green's function
    DMFT_SCC(params['Delta'][0])

    # Running ctqmc
    print('Running ---- qmc itt.: ', it, '-----')
    subprocess.call(params['exe'][0], shell=True,stdout=sys.stdout,stderr=sys.stderr)
    
    # Some copying to store data obtained so far (at each iteration)
    cmd = 'cp Gf.out '+dir+'Gf.out.'+str(it)
    subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr)  # copying Gf
    cmd = 'cp Sig.out '+dir+'Sig.out.'+str(it)
    subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr) # copying Sig
    cmd = 'cp ctqmc.log '+dir+'ctqmc.log.'+str(it)
    subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr) # copying log file
    cmd = 'cp rDelta.tau.00.000 '+dir+'rDelta.tau.00.000.'+str(it)
    subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr) # copying
    cmd = 'cp rDelta.tau.01.000 '+dir+'rDelta.tau.01.000.'+str(it)
    subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr) # copying
    cmd = 'cp Delta.tau.00.000 '+dir+'Delta.tau.00.000.'+str(it)
    subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr) # copying 
    cmd = 'cp Delta.tau.01.000 '+dir+'Delta.tau.01.000.'+str(it)
    subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr) # copying

    
    
    if it>0:
        diff = Diff('Gf.out', '{}Gf.out.'.format(dir)+str(it-1))
        print('Diff=', diff)

subprocess.call('rm ctqmc.log Delta.inp Deltat.inp Gf.out PARAMS PPSigma.OCA Sig.out Delta.tau.00.000 Delta.tau.01.000 rDelta.tau.01.000 rDelta.tau.00.000', shell=True) 
for i in np.arange(8):
    subprocess.call('rm status.00{}'.format(i), shell=True)
        