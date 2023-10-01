#!/usr/bin/env python
# Author: Kristjan Haule, March 2007-2017
from scipy import * 
import os,sys,subprocess
import numpy as np
import matplotlib.pyplot as plt 
from scipy import integrate
import hilbert
import perturb_lib
"""
This module runs ctqmc impurity solver for one-band model.
The executable should exist in directory params['exe']
This controls the workflow of dmft.
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
if (len(sys.argv)==4):
    Uc=float(sys.argv[1])
    T=float(sys.argv[2])
    mode=float(sys.argv[3])#0: no perturbation. 1: with perturbation
    print('T=',T)
    print('Uc=',Uc)
    print('mode=',mode)
if mode==0:
    dir='./files_ctqmc/{}_{}/'.format(Uc,T)
elif mode==1:
    dir='./files_pert_ctqmc/{}_{}/'.format(Uc,T)

if os.path.exists(dir):
    print('already have this directory: ', dir)
    # CAREFULL! delete any previous files
    subprocess.call('rm '+dir+'*', shell=True) 
else:
    print('directory does not exist... make a new one')
    cmd_newfolder='mkdir '+dir
    subprocess.call(cmd_newfolder, shell=True)
    
subprocess.call('rm ctqmc.log Delta.inp Deltat.inp Gf.out PARAMS PPSigma.OCA Sig.out Delta.tau.00.000 Delta.tau.01.000 rDelta.tau.01.000 rDelta.tau.00.000 sig12.dat', shell=True) 
subprocess.call('rm Aw.out.001 Aw.out.000 Gcoeff.dat gs_qmc.dat Sig.outB Sig.outD Sw.dat Probability.dat Gt.dat Gw.dat histogram.dat nohup_imp.out.000 ctqmc.log Delta.inp Deltat.inp Gf.out PARAMS PPSigma.OCA Sig.out Delta.tau.00.000 Delta.tau.01.000 rDelta.tau.01.000 rDelta.tau.00.000 ori_Delta.inp', shell=True) 
for i in np.arange(10):
    subprocess.call('rm status.00{}'.format(i), shell=True)
subprocess.call('rm status.010', shell=True)
subprocess.call('rm status.011', shell=True)
# for i in np.arange(8):
#     subprocess.call('rm status.00{}'.format(i), shell=True)
   

params = {"exe":   ["mpirun -np 8 ./ctqmc",          "# Path to executable"],
          "U":     [Uc,                 "# Coulomb repulsion (F0)"],
          "mu":    [Uc/2.,              "# Chemical potential"],
          "beta":  [1/T,                "# Inverse temperature"],
          "M" :    [2e7,                "# Number of Monte Carlo steps"],
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

def DMFT_SCC(fDelta,opt=0):
    """This subroutine creates Delta.inp from Gf.out for DMFT on bethe lattice: Delta=t^2*G
    If Gf.out does not exist, it creates Gf.out which corresponds to the non-interacting model
    In the latter case also creates the inpurity cix file, which contains information about
    the atomic states.
    """
    fileGf = 'Gf.out'
    # filesig='./trial_sigma/{}_{}.dat'.format(Uc,T)# use this filesig, DMFT will start repeatedly from a same trial sigma, to test if it is stable.
    filesig='Sig.out'
    #get sigma
    if (os.path.exists(filesig)): 
        sigma = np.loadtxt(filesig)
    else:# generate a trial sigma
        print("trial sigma cannot found...preparing trial sigma...")
        # return 0
        sigma=np.zeros((500,5),dtype=complex)
        for i in np.arange(500):
            omega=(2*i+1)*np.pi/params['beta'][0]
            sigma[i,0]=omega #omega
            sigma[i,1]=params['mu'][0]+0.01# Gaa,up,real
            sigma[i,2]=0# Gaa, up, imag
            sigma[i,3]=params['mu'][0]-0.01# Gaa, dn, real
            sigma[i,4]=0# Gaa, dn, imag
        print('writing trial sigma file')
        f = open(filesig, 'w')
        for i in range(np.shape(sigma)[0]):# consider the case that we may have many columns of Gf
            # print(Gf[i,0], 0.25*Gf[i,1], 0.25*Gf[i,2], file=f) # This is DMFT SCC: Delta = t**2*G (with t=1/2)
            # print(np.shape(Deltafile))
            for k in np.arange(np.shape(sigma)[1]):
                print(sigma[i,k].real,end='\t', file=f)# print in the same line
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
    freq_num=500
    om = sigma[:freq_num,0].real
    Sg_A = sigma[:freq_num,1]+sigma[:freq_num,2]*1j
    Sg_B = sigma[:freq_num,3]+sigma[:freq_num,4]*1j
    # print(type(Sg_A),type(om),type(params['mu'][0]))

    #also, prepare the original delta for comparison.
    if opt==0:# no perturbation
        # ori_Dlt_A,ori_Dlt_B = hilbert.SCC_AFM(W, om, params['beta'][0], params['mu'][0], params['U'][0], Sg_A, Sg_B, False)
        # ori_Dlt_A,ori_Dlt_B =perturb.Delta_DMFT(Sg_A,Sg_B,Uc,T,20)# corrected delta. this is easy so we can use denser k points.
        ori_Dlt_A,ori_Dlt_B =perturb_lib.Delta_DMFT( Sg_A, Sg_B,Uc,T)

        # plt.plot(ori_Dlt_A.real,label='delta11 real')
        # plt.plot(ori_Dlt_A.imag,label='delta11 imag')
        # plt.plot(ori_Dlt_A_HT.real,label='delta11_HT real')
        # plt.plot(ori_Dlt_A_HT.imag,label='delta11_HT imag')
        # plt.legend()
        # plt.grid()
        # plt.show()
        f = open(fDelta, 'w')
        for i,iom in enumerate(om):
            print(iom, ori_Dlt_A[i].real, ori_Dlt_A[i].imag, ori_Dlt_B[i].real, ori_Dlt_B[i].imag, file=f) 
        f.close()
        cmd = 'cp Delta.inp '+dir+'ori_Delta.inp.'+str(it)
        subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr)  # copying Gf
    elif opt==1:#perturbation

        # use this line to run original DMFT.
        # Dlt_A,Dlt_B=perturb.Delta_pert_DMFT(Sg_A,Sg_B,Uc,T,10)
        # Preparing input file Delta.inp
        cmd_pert='mpirun -np 8 python perturb.py {} {} {}'.format(Uc,T,filesig)
        subprocess.call(cmd_pert, shell=True)
        cmd = 'cp Delta.inp '+dir+'pert_Delta.inp.'+str(it)
        subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr)  # copying Gf
    print('Delta file is done!')


def Diff(fg1, fg2):
    data1 = np.loadtxt(fg1).transpose()
    data2 = np.loadtxt(fg2).transpose()
    diff = np.sum(abs(data1-data2))/(np.shape(data1)[0]*np.shape(data1)[1])
    return diff

# Number of DMFT iterations
Niter = 20
diff_arr=np.zeros(Niter)
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
    # note: first run the non-pert version and get self-energy for reference.
    # Then,calculate pert-version. then the final sig.out left is the sig function after perturbation.
    if mode ==0:
        DMFT_SCC(params['Delta'][0],0)# DMFT without perturbation
            # Running ctqmc
        print('Running ---- qmc itt.: ', it, '-----')
        subprocess.call(params['exe'][0], shell=True,stdout=sys.stdout,stderr=sys.stderr)
    
        # Some copying to store data obtained so far (at each iteration)
        cmd = 'cp Gf.out '+dir+'ori_Gf.out.'+str(it)
        subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr)  # copying Gf
        cmd = 'cp Sig.out '+dir+'ori_Sig.out.'+str(it)
        subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr) # copying Sig
        cmd = 'cp ctqmc.log '+dir+'ori_ctqmc.log.'+str(it)
        subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr) # copying log file
        cmd = 'cp rDelta.tau.00.000 '+dir+'ori_rDelta.tau.00.000.'+str(it)
        subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr) # copying
        cmd = 'cp rDelta.tau.01.000 '+dir+'ori_rDelta.tau.01.000.'+str(it)
        subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr) # copying
        cmd = 'cp Delta.tau.00.000 '+dir+'ori_Delta.tau.00.000.'+str(it)
        subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr) # copying 
        cmd = 'cp Delta.tau.01.000 '+dir+'ori_Delta.tau.01.000.'+str(it)
        subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr) # copying




    if mode ==1:# if specified, do pert DMFT. otherwise, don't do.
        if it==0:
            DMFT_SCC(params['Delta'][0],0)
        else:
            DMFT_SCC(params['Delta'][0],1)# DMFT with perturbation

        # Running ctqmc
        print('Running ---- pert qmc itt.: ', it, '-----')
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
        cmd = 'cp sig12.dat '+dir+'sig12.dat.'+str(it)
        subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr) # copying sig12 file


    
    if it>0:
        if mode==1:
            diff = Diff('Gf.out', '{}Gf.out.'.format(dir)+str(it-1))
        elif mode==0:
            diff = Diff('Gf.out', '{}ori_Gf.out.'.format(dir)+str(it-1))
        print('Diff=', diff)
        diff_arr[it-1]=diff
        if (diff<1e-6): break
        
diff_dir=dir+'diff.log'
f = open(diff_dir, 'w')
for iter in np.arange(it):
    print(diff_arr[iter], file=f) 
f.close()

subprocess.call('rm Aw.out.001 Aw.out.000 sig12.dat Gcoeff.dat gs_qmc.dat Sig.outB Sig.outD Sw.dat Probability.dat Gt.dat Gw.dat histogram.dat nohup_imp.out.000 ctqmc.log Delta.inp Deltat.inp Gf.out PARAMS PPSigma.OCA Sig.out Delta.tau.00.000 Delta.tau.01.000 rDelta.tau.01.000 rDelta.tau.00.000 ori_Delta.inp', shell=True) 
for i in np.arange(8):
    subprocess.call('rm status.00{}'.format(i), shell=True)
        