import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess
import time
import hilbert
# this code is written to reproduce a few leading order of 





# from DMFT, we only have first 2000 positive Matsubara freqs.
# hoeever we need to extend our sigma to 4 times as before.
# The structure is:
#|          first 2000                |     second 2000        |     third 2000        | last 2000  |
#| high freq behavior at negative side|  negative-frequencies  |  original sigma data  |  sigma at high freq which is estimated using high-freq behavior  |
def ext_sig(sig,beta):
    lenom=sig.size
    # print(lenom)
    all_om=(2*np.arange(2*lenom)+1)*np.pi/beta
    allsig=np.zeros(4*lenom,dtype=complex)
    allsig[2*lenom:3*lenom]=sig
    allsig[3*lenom:4*lenom]=sig[lenom-1].real+1j*sig[lenom-1].imag*all_om[lenom-1]/all_om[lenom:2*lenom]
    allsig[:2*lenom]=allsig[4*lenom:2*lenom-1:-1].conjugate()
    return allsig

def fermi(eps,beta):
    return 1/(np.exp(beta*eps)+1)
                         


def pertimp():
    T=0.01
    U=2.0
    dosfile='DOS_3D.dat'
    sigfile='{}_{}.dat'.format(U,T)
    mu=U/2.
    beta=1/T
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
    
    #calculate P. P is calculated on Boson matsubara freq points!
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
    

def pertimp_func(G_A,G_B,delta_inf,beta,U,eps2_ave):
    T=1/beta
    lenom=int(G_A.size/4)
    n=lenom
    iom= 1j*(2*np.arange(4*n)+1-4*n)*np.pi/beta
    iOm= 1j*(2*np.arange(2*n+1)-2*n)*np.pi/beta
    G_A0=(iom+delta_inf)/(iom**2-delta_inf**2-eps2_ave)
    # G_B0=(iom-delta_inf)/(iom**2-delta_inf**2-eps2_ave)
    # print('n=',lenom)
    # check the generated impurity green function!
    # plt.plot(G_A.real,label='Gimp_A real')
    # plt.plot(G_A.imag,label='Gimp_A imag')
    # plt.plot(G_B.real,label='Gimp_B real')
    # plt.plot(G_B.imag,label='Gimp_B imag')
    # plt.plot(G_A0.real,label='Gimp_A0 real')
    # plt.plot(G_A0.imag,label='Gimp_A0 imag')
    # plt.plot(G_B0.real,label='Gimp_B0 real')
    # plt.plot(G_B0.imag,label='Gimp_B0 imag')
    # plt.legend()
    # plt.show()

    #calculate P. P is calculated on Boson matsubara freq points!
    P_A=np.zeros(lenom*2+1,dtype=complex)
    P_B=np.zeros(lenom*2+1,dtype=complex)
    alpha_ave=np.sqrt(eps2_ave+delta_inf**2)
    lindhard1=0.25*(1-delta_inf**2/alpha_ave**2)*(2*fermi(alpha_ave,beta)-1)/(iOm+2*alpha_ave)
    lindhard2=0.25*(1-delta_inf**2/alpha_ave**2)*(1-2*fermi(alpha_ave,beta))/(iOm-2*alpha_ave)
    lindhard=lindhard1+lindhard2
    for i in np.arange(lenom*2+1):# here, i is the index of boson matsubara freq.
        # lindhard1=0.25*(1-delta_inf**2/alpha_ave**2)*(2*fermi(alpha_ave)-1)/(iOm[i]+2*alpha_ave)
        # lindhard2=0.25*(1-delta_inf**2/alpha_ave**2)*(1-2*fermi(alpha_ave))/(iOm[i]-2*alpha_ave)
        P_A[i]=lindhard[i]+T*np.sum(G_A[lenom:lenom*3]*G_A[lenom+i-lenom:lenom*3+i-lenom]-G_A0[lenom:lenom*3]*G_A0[lenom+i-lenom:lenom*3+i-lenom])#
        # P_B[i]=T*np.sum(G_B[lenom:lenom*3]*G_B[lenom+i-lenom:lenom*3+i-lenom]-G_B0[lenom:lenom*3]*G_B0[lenom+i-lenom:lenom*3+i-lenom])#
    P_B=P_A
    # take a look at P
    # plt.plot(P_A.real,label='Pimp_A real')
    # plt.plot(P_A.imag,label='Pimp_A imag')
    # plt.plot(P_B.real,label='Pimp_B real')
    # plt.plot(P_B.imag,label='Pimp_B imag')
    # plt.legend()
    # plt.show()
    
    #calculate sig. sig is calculate on fermion matsubara freq points!
    sigp_A=np.zeros(lenom*2,dtype=complex)
    sigp_B=np.zeros(lenom*2,dtype=complex)
    for i in np.arange(lenom*2):# here, i is the index of fermion matsubara freq.
        sigp_A[i]=-U**2*T*np.sum(P_B*G_A[lenom+i-lenom:lenom*3+i-lenom+1])
        sigp_B[i]=-U**2*T*np.sum(P_A*G_B[lenom+i-lenom:lenom*3+i-lenom+1])
    plt.plot(sigp_A.real,label='sig(imp,2)_A real')
    plt.plot(sigp_A.imag,label='sig(imp,2)_A imag')
    plt.plot(sigp_B.real,label='sig(imp,2)_B real')
    plt.plot(sigp_B.imag,label='sig(imp,2)_B imag')
    # plt.legend()
    # plt.show()
    return sigp_A,sigp_B





# pertimp()

