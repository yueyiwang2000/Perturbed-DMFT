import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess
import time
import hilbert
# this code is written to reproduce a few leading order of 



T=0.01
U=2.0
dosfile='DOS_3D.dat'
sigfile='sigtest_{}_{}.dat'.format(U,T)
mu=U/2.
beta=1/T

# from DMFT, we only have first 2000 positive Matsubara freqs.
# hoeever we need to extend our sigma to 4 times as before.
# The structure is:
#|          first 2000                |     second 2000        |     third 2000        | last 2000  |
#| high freq behavior at negative side|  negative-frequencies  |  original sigma data  |  sigma at high freq which is estimated using high-freq behavior  |
def ext_sig(sig):
    lenom=sig.size
    # print(lenom)
    all_om=(2*np.arange(2*lenom)+1)*np.pi/beta
    allsig=np.zeros(4*lenom,dtype=complex)
    allsig[2*lenom:3*lenom]=sig
    allsig[3*lenom:4*lenom]=sig[lenom-1].real+1j*sig[lenom-1].imag*all_om[lenom-1]/all_om[lenom:2*lenom]
    allsig[:2*lenom]=allsig[4*lenom:2*lenom-1:-1].conjugate()
    return allsig


                         


def repro_Brute_Force():
    # read dos and generate Hilbert transformation
    x, Di = np.loadtxt(dosfile).T
    # plt.plot(x,Di)
    # plt.show()
    W = hilbert.Hilb(x,Di)

    sigma=np.loadtxt(sigfile)
    om=sigma[:,0]
    lenom=om.size
    print('# of frequencies: ',lenom)
    allom=(2*np.arange(4*lenom)+1-lenom*4)*np.pi/beta
    if om[0]/(np.pi/beta)>1.01 or om[0]/(np.pi/beta)<0.99:
        print('seems the temperature does not match!')
        return 0
    sigA_short=sigma[:,1]+1j*sigma[:,2]
    sigB_short=sigma[:,3]+1j*sigma[:,4]
    sigA=ext_sig(sigA_short)
    sigB=ext_sig(sigB_short)
    # # take a glance of my sigma!
    # plt.plot(sigA.real,label='sigA real')
    # plt.plot(sigA.imag,label='sigA imag')
    # plt.plot(sigB.real,label='sigB real')
    # plt.plot(sigB.real,label='sigB imag')
    # plt.legend()
    # plt.show()

    G_A=np.zeros_like(sigA,dtype=complex)
    G_B=np.zeros_like(sigB,dtype=complex)
    for i in np.arange(4*lenom):
        z_A=1j*allom[i]+mu-sigA[i]
        z_B=1j*allom[i]+mu-sigB[i]
        G_A[i]=W(z_A)
        G_B[i]=W(z_B)
    # check the generated impurity green function!
    plt.plot(G_A.real,label='G_A real')
    plt.plot(G_A.imag,label='G_A imag')
    plt.plot(G_B.real,label='G_B real')
    plt.plot(G_B.imag,label='G_B imag')
    plt.legend()
    plt.show()
    
    #calculate P. P is calculate on Boson matsubara freq points!
    P_A=np.zeros(lenom*2,dtype=complex)
    P_B=np.zeros(lenom*2,dtype=complex)
    for i in np.arange(lenom*2):# here, i is the index of boson matsubara freq.
        P_A[i]=T*np.sum(G_A[lenom:lenom*3]*G_A[lenom+i-lenom:lenom*3+i-lenom])
        P_B[i]=T*np.sum(G_B[lenom:lenom*3]*G_B[lenom+i-lenom:lenom*3+i-lenom])
    # take a look at P
    plt.plot(P_A.real,label='P_A real')
    plt.plot(P_A.imag,label='P_A imag')
    plt.plot(P_B.real,label='P_B real')
    plt.plot(P_B.imag,label='P_B imag')
    plt.legend()
    plt.show()
    
    #calculate sig. sig is calculate on fermion matsubara freq points!
    sigp_A=np.zeros(lenom*2,dtype=complex)
    sigp_B=np.zeros(lenom*2,dtype=complex)
    for i in np.arange(lenom*2):# here, i is the index of fermion matsubara freq.
        sigp_A[i]=-U**2*T*np.sum(P_B*G_A[lenom+i-lenom:lenom*3+i-lenom])
        sigp_B[i]=-U**2*T*np.sum(P_A*G_B[lenom+i-lenom:lenom*3+i-lenom])
    plt.plot(sigp_A.real,label='sigp_A real')
    plt.plot(sigp_A.imag,label='sigp_A imag')
    plt.plot(sigp_B.real,label='sigp_B real')
    plt.plot(sigp_B.imag,label='sigp_B imag')
    plt.legend()
    plt.show()
    # corrected sigma
    plt.plot(sigA_short[-1].real+sigp_A.real,label='sig_bf_pert_A real')
    plt.plot(sigp_A.imag,label='sig_bf_pert_A imag')
    plt.plot(sigB_short[-1].real+sigp_B.real,label='sig_bf_pert_B real')
    plt.plot(sigp_B.imag,label='sig_bf_pert_B imag')
    plt.plot(sigA[lenom:lenom*3].real,label='sigA real')
    plt.plot(sigA[lenom:lenom*3].imag,label='sigA imag')
    plt.plot(sigB[lenom:lenom*3].real,label='sigB real')
    plt.plot(sigB[lenom:lenom*3].real,label='sigB imag')
    plt.legend()
    plt.show()
    return 0 
# sigtest=np.array([1+1j,2+2j,3+3j,4+4j])
# print(ext_sig(sigtest))
    
repro_Brute_Force()

