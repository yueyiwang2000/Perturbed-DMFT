import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess
import time
from joblib import Parallel, delayed

def dispersion(kx,ky,kz,a=1,t=1):
    e_k=-2*t*np.cos(kx*a)-2*t*np.cos(ky*a)-2*t*np.cos(kz*a)
    return e_k


def foldback(k,knum):# from complete 2*knum k points per dimension to a reduced knum k points.
    return np.where((k>=0)&(k<knum),k,2*knum-k-1)# 
    
def gen_qindices(qlist):
    qindices=[]
    for i in qlist:
        for j in qlist:
            for k in qlist:
                if i<=j<=k:
                    qindices+=[(i,j,k)]
    # print(qindices)
    return qindices


# Use this, we want to generate an array G_sk(iomega) as a function of iomega.
# instead we prepare alpha and then generate Gf in the polarization
def z(beta,mu,sig):
    # sometimes we want values of G beyond the range of n matsubara points. try to do a simple estimation for even higher freqs:
    n=sig.size
    om=(2*np.arange(4*n)+1-4*n)*np.pi/beta
    allsig=ext_sig(beta,sig)
    z=om*1j+mu-allsig
    return z

def z0(beta,mu,allsig):# Here, alpha=i*omega+mu-sig(inf).real
    n=int(allsig.size/4)
    om=(2*np.arange(4*n)+1-4*n)*np.pi/beta
    # allsig=ext_sig(beta,sig)
    z0=om*1j+mu-allsig[-1].real#
    return z0

def ext_sig(beta,sig):
    lenom=sig.size
    # print(lenom)
    all_om=(2*np.arange(2*lenom)+1)*np.pi/beta
    allsig=np.zeros(4*lenom,dtype=complex)
    allsig[2*lenom:3*lenom]=sig
    allsig[3*lenom:4*lenom]=sig[lenom-1].real+1j*sig[lenom-1].imag*all_om[lenom-1]/all_om[lenom:2*lenom]
    allsig[:2*lenom]=allsig[4*lenom:2*lenom-1:-1].conjugate()
    return allsig

def G_diag_A(knum,z_A,z_B,a=1):
    kall=np.linspace(-np.pi/a,np.pi/a,num=knum+1)
    kroll=np.roll(kall,1)
    kave=(kall+kroll)/2
    klist=kave[-knum:] 
    n=z_A.size
    k1, k2, k3 = np.meshgrid(klist, klist, klist, indexing='ij')
    kx=0.5*(-k1+k2+k3)
    ky=0.5*(k1-k2+k3)
    kz=0.5*(k1+k2-k3)
    G_diag_A=np.zeros((n,knum,knum,knum),dtype=np.complex128)
    zazb=z_A*z_B
    G_diag_A = z_B[:, None, None, None] / (zazb[:, None, None, None] - dispersion(kx, ky, kz)**2)
    return G_diag_A

def G_diag_B(knum,z_A,z_B,a=1):
    kall=np.linspace(-np.pi/a,np.pi/a,num=knum+1)
    kroll=np.roll(kall,1)
    kave=(kall+kroll)/2
    klist=kave[-knum:] 
    n=z_A.size
    k1, k2, k3 = np.meshgrid(klist, klist, klist, indexing='ij')
    kx=0.5*(-k1+k2+k3)
    ky=0.5*(k1-k2+k3)
    kz=0.5*(k1+k2-k3)
    G_diag_B=np.zeros((n,knum,knum,knum),dtype=np.complex128)
    zazb=z_A*z_B
    G_diag_B = z_A[:, None, None, None] / (zazb[:, None, None, None] - dispersion(kx, ky, kz)**2)
    return G_diag_B

def G_offdiag(knum,z_A,z_B,a=1):
    kall=np.linspace(-np.pi/a,np.pi/a,num=knum+1)
    kroll=np.roll(kall,1)
    kave=(kall+kroll)/2
    klist=kave[-knum:] 
    n=z_A.size
    k1, k2, k3 = np.meshgrid(klist, klist, klist, indexing='ij')
    kx=0.5*(-k1+k2+k3)
    ky=0.5*(k1-k2+k3)
    kz=0.5*(k1+k2-k3)
    G_offdiag=np.zeros((n,knum,knum,knum),dtype=np.complex128)
    zazb=z_A*z_B
    dis=dispersion(kx, ky, kz)
    G_offdiag = dis / (zazb[:, None, None, None] - dis**2)
    return G_offdiag

def G_diagonalized(knum,allsig,beta,mu,a=1):
    #setting k grid
    n=int(allsig.size/4)
    kall=np.linspace(-np.pi/a,np.pi/a,num=knum+1)
    kroll=np.roll(kall,1)
    kave=(kall+kroll)/2
    klist=kave[-knum:] 
    k1, k2, k3 = np.meshgrid(klist, klist, klist, indexing='ij')
    kx=0.5*(-k1+k2+k3)
    ky=0.5*(k1-k2+k3)
    kz=0.5*(k1+k2-k3)
    #
    om=(2*np.arange(4*n)+1-4*n)*np.pi/beta
    z_bar=1j*(om-allsig.imag)# z_bar is imaginary.
    delta=allsig.real-mu# we don't have to care the sign of delta.
    G_pp=np.zeros((n,knum,knum,knum),dtype=np.complex128)
    G_pp = 1 / (z_bar[:, None, None, None] + np.sqrt(delta[:, None, None, None]**2+dispersion(kx, ky, kz)**2))
    G_mm=np.zeros((n,knum,knum,knum),dtype=np.complex128)
    G_mm = 1 / (z_bar[:, None, None, None] - np.sqrt(delta[:, None, None, None]**2+dispersion(kx, ky, kz)**2))
    return G_pp,G_mm

def precalcG0(knum,z_0,a=1):
    k1=np.linspace(-np.pi/a,np.pi/a,num=knum+1)
    k2=np.roll(k1,1)
    kave=(k1+k2)/2
    klist=kave[-knum:] 
    n=z_0.size
    k1, k2, k3 = np.meshgrid(klist, klist, klist, indexing='ij')
    kx=0.5*(-k1+k2+k3)
    ky=0.5*(k1-k2+k3)
    kz=0.5*(k1+k2-k3)
    n=z_0.size
    kx, ky, kz = np.meshgrid(klist, klist, klist, indexing='ij')
    G_k=np.zeros((n,knum,knum,knum),dtype=np.complex128)
    G_k = 1 / (z_0[:, None, None, None] - dispersion(kx, ky, kz))
    return G_k

#here, I specify Omega_index is index of boson freq, as it should be.
# and, let's set qx as sth like [1/10,3/10,5/10,7/10,9/10]. half as the k_f
# this means, qx[0]=1/10. This is an eg in knum=10.

def precalcP_innerloop(q, kxind, kyind, kzind, knum, n, G1,G2):
    qx, qy, qz = q
    P_partial = np.zeros((2*n+1, 1, 1, 1), dtype=complex)
    G2_kq = G2[:, foldback(kxind + qx, knum), foldback(kyind + qy, knum), foldback(kzind + qz, knum)]
    for Omind in np.arange(n+n+1):
        P_partial[Omind, 0, 0, 0] = np.sum(G1[n:3*n] * G2_kq[n+Omind-n:3*n+Omind-n]) 
        # P_partial[2*n-Omind, 0, 0, 0]=P_partial[Omind, 0, 0, 0]   
    return P_partial

def precalcP(beta, knum, G1,G2, a=1):
    n = int(np.shape(G1)[0] / 4)
    kind_list = np.arange(knum)
    if knum % 2 != 0:
        print('knum should be a even number!')
        return 0
    halfknum = int(knum / 2)
    qind_list = np.arange(halfknum+1)
    P = np.zeros((2*n+1, halfknum+1, halfknum+1, halfknum+1), dtype=complex)
    # fermion_Omega_ind = np.arange(n)
    kxind, kyind, kzind = np.meshgrid(kind_list, kind_list, kind_list, indexing='ij')

    # Flatten the qind_list
    # q_indices = [(qx, qy, qz) for qx in qind_list for qy in qind_list for qz in qind_list]
    q_indices=gen_qindices(qind_list)
    # Parallelize the inner loop
    results = Parallel(n_jobs=-1)(delayed(precalcP_innerloop)(q, kxind, kyind, kzind, knum, n, G1,G2) for q in q_indices)
    # Combine the results
    for i, q in enumerate(q_indices):
        qx, qy, qz = q
        P[:, qx, qy, qz] = np.squeeze(results[i])
        P[:, qx, qz, qy] = np.squeeze(results[i])
        P[:, qy, qx, qz] = np.squeeze(results[i])
        P[:, qy, qz, qx] = np.squeeze(results[i])
        P[:, qz, qx, qy] = np.squeeze(results[i])
        P[:, qz, qy, qx] = np.squeeze(results[i])
    return P / beta * (1/ a / knum) ** 3#2 * np.pi 

def fermi(eps,beta):
    return 1/(np.exp(beta*eps)+1)

def precalcP_innerloop_improved(q, kxind, kyind, kzind, knum, n, G_k,G_k0,alphak,f_alphak,beta):
    qx, qy, qz = q
    P_partial = np.zeros((2*n+1, 1, 1, 1), dtype=complex)
    G_kq = G_k[:, foldback(kxind + qx, knum), foldback(kyind + qy, knum), foldback(kzind + qz, knum)]
    G_kq0 = G_k0[:, foldback(kxind + qx, knum), foldback(kyind + qy, knum), foldback(kzind + qz, knum)]
    alpha_kq=alphak[foldback(kxind + qx, knum), foldback(kyind + qy, knum), foldback(kzind + qz, knum)]
    falpha_kq=f_alphak[foldback(kxind + qx, knum), foldback(kyind + qy, knum), foldback(kzind + qz, knum)]
    
    for Omind in np.arange(n+1)+n:
        lindhard=(f_alphak-falpha_kq)/(1j*(Omind-n-0.001)*2*np.pi/beta-alpha_kq+alphak)
        P_partial[Omind, 0, 0, 0] =  +beta*np.sum(lindhard)+np.sum(G_k[n:3*n] * G_kq[n+Omind-n:3*n+Omind-n]-G_k0[n:3*n] * G_kq0[n+Omind-n:3*n+Omind-n])
        P_partial[2*n-Omind, 0, 0, 0]=P_partial[Omind, 0, 0, 0].conjugate()
    return P_partial

def precalcP_improved(beta, knum, Gk, fullsig,mu,opt=0,a=1):# here, we need original sigma.
    n = int(np.shape(Gk)[0] / 4)
    kind_list = np.arange(knum) 
    if knum % 2 != 0:
        print('knum should be a even number!')
        return 0
    halfknum = int(knum / 2)
    qind_list = np.arange(halfknum+1)
    #prepare sth for this trick.


    # generate a klist and a kqlist
    k1=np.linspace(-np.pi/a,np.pi/a,num=knum+1)
    k2=np.roll(k1,1)
    kave=(k1+k2)/2
    klist=kave[-knum:] 
    # create a 3D array alpha=eps_k-mu+sigma(inf)
    k1, k2, k3 = np.meshgrid(klist, klist, klist, indexing='ij')
    kx=0.5*(-k1+k2+k3)
    ky=0.5*(k1-k2+k3)
    kz=0.5*(k1+k2-k3)

    #generate alpha, f(alpha) for freq sum
    delta_inf=-mu+fullsig[-1].real
    if opt=='up':
        alphak=-np.sqrt(dispersion(kx,ky,kz)**2+delta_inf**2)
    elif opt=='dn':
        alphak=np.sqrt(dispersion(kx,ky,kz)**2+delta_inf**2)
    else:
        print('please specify up or dn!')
        return 0
    f_alphak=fermi(alphak,beta)
    #generate unperturbed Green's function

    # z_0=z0(beta,mu,fullsig)
    om=(2*np.arange(4*n)+1-4*n)*np.pi/beta
    z_bar=1j*(om)# z_bar is imaginary.-fullsig.imag
    Gk0=np.zeros((n,knum,knum,knum),dtype=np.complex128)
    if opt=='up':
        Gk0 = 1 / (z_bar[:, None, None, None] + np.sqrt(delta_inf**2+dispersion(kx, ky, kz)**2))
    elif opt=='dn':
        Gk0 = 1 / (z_bar[:, None, None, None] - np.sqrt(delta_inf**2+dispersion(kx, ky, kz)**2))
    
    # Gk0=precalcG0(knum,z_0)


    P = np.zeros((2*n+1, halfknum+1, halfknum+1, halfknum+1), dtype=complex)
    # print('shape of P,',np.shape(P))
    # fermion_Omega_ind = np.arange(n)
    kxind, kyind, kzind = np.meshgrid(kind_list, kind_list, kind_list, indexing='ij')

    # Flatten the qind_list
    # q_indices = [(qx, qy, qz) for qx in qind_list for qy in qind_list for qz in qind_list]
    q_indices=gen_qindices(qind_list)
    # Parallelize the inner loop
    results = Parallel(n_jobs=-1)(delayed(precalcP_innerloop_improved)(q, kxind, kyind, kzind, knum, n, Gk,Gk0,alphak,f_alphak,beta) for q in q_indices)
    for i, q in enumerate(q_indices):
        qx, qy, qz = q
        P[:, qx, qy, qz] = np.squeeze(results[i])
        P[:, qx, qz, qy] = np.squeeze(results[i])
        P[:, qy, qx, qz] = np.squeeze(results[i])
        P[:, qy, qz, qx] = np.squeeze(results[i])
        P[:, qz, qx, qy] = np.squeeze(results[i])
        P[:, qz, qy, qx] = np.squeeze(results[i])
    return P / beta * (1/ a / knum) ** 3

def boundary_modification(P):#give a 1/2 factor to those points on the boundary.
    new_P=P
    # print('P:',P[0,-1,-1,-1])
    new_P[:,-1,:,:]=new_P[:,-1,:,:]/2
    new_P[:,:,-1,:]=new_P[:,:,-1,:]/2
    new_P[:,:,:,-1]=new_P[:,:,:,-1]/2
    # print('newP:',new_P[0,-1,-1,-1])
    return new_P


# Note: remember G and P has different spin.
#since every sigma_k can be only used once, seems we don't need to store them in RAM.
# However, for a specific k point, we still want sigma at every fermion matsubara freq.

# as polarization function, I specify omega_index is index of both fremion and boson freq.

def precalcsig_innerloop(k, qxind, qyind, qzind, knum, n, P_k, Gk):
    kx, ky, kz = k
    sig_partial = np.zeros((2*n, 1, 1, 1), dtype=complex)
    G_kq = Gk[:, foldback(qxind + kx, knum), foldback(qyind + ky, knum), foldback(qzind + kz, knum)]
    for omind in np.arange(2*n):
        sig_partial[omind, 0, 0, 0] = np.sum(P_k * G_kq[omind:omind +2*n+1])# from omind, omind+1, ..., to omind+2n
    return sig_partial

def precalcsig(U,beta, knum, Pk, Gk, a=1):
    n = int(np.shape(Gk)[0]/4)
    # print('n=',n)
    
    if knum % 2 != 0:
        print('knum should be a even number!')
        return 0
    halfknum = int(knum / 2)
    qind_list = np.arange(halfknum+1)# grid of P
    fullqind_list=np.arange(knum+1)
    kind_list=np.arange(halfknum)
    newP=boundary_modification(Pk)
    # print(np.concatenate((kind_list[::-1],kind_list)))
    inv_qlist=qind_list[::-1]
    fulllist_P=np.concatenate((inv_qlist[:-1],qind_list))
    # print('fullklist_P',fulllist_P)
    fullindx,fullindy,fullindz=np.meshgrid(fulllist_P,fulllist_P,fulllist_P,indexing='ij')
    modified_Pk=newP[:,fullindx,fullindy,fullindz]
    # print('shape of entire P:',np.shape(modified_Pk))
    sig = np.zeros((2*n, halfknum, halfknum, halfknum), dtype=complex)
    # fermion_Omega_ind = np.arange(n)
    qxind, qyind, qzind = np.meshgrid(fullqind_list, fullqind_list, fullqind_list, indexing='ij')

    # Flatten the qind_list
    # k_indices = [(kx, ky, kz) for kx in kind_list for ky in kind_list for kz in kind_list]
    k_indices=gen_qindices(kind_list)
    # Parallelize the inner loop
    results = Parallel(n_jobs=-1)(delayed(precalcsig_innerloop)(k, qxind, qyind, qzind, knum, n, modified_Pk, Gk) for k in k_indices)
    # Combine the results
    for i, k in enumerate(k_indices):
        kx, ky, kz = k
        sig[:, kx, ky, kz] = np.squeeze(results[i])
        sig[:, kx, kz, ky] = np.squeeze(results[i])
        sig[:, ky, kx, kz] = np.squeeze(results[i])
        sig[:, ky, kz, kx] = np.squeeze(results[i])
        sig[:, kz, kx, ky] = np.squeeze(results[i])
        sig[:, kz, ky, kx] = np.squeeze(results[i])
    return sig*-1*U*U / beta * ( 1/ a / knum) ** 3#2*np.pi

def reverse_diag(knum,sigmapp,impsig,mu,a=1):
    # n = int(np.shape(sigmapp)[0])
    kall=np.linspace(-np.pi/a,np.pi/a,num=knum+1)
    kroll=np.roll(kall,1)
    kave=(kall+kroll)/2
    klist=kave[-knum:] 
    halfknum=int(knum/2)
    # think! did I use the correct part of klist?
    k1, k2, k3 = np.meshgrid(klist, klist, klist, indexing='ij')
    kx=0.5*(-k1+k2+k3)
    ky=0.5*(k1-k2+k3)
    kz=0.5*(k1+k2-k3)
    disp=dispersion(kx,ky,kz)
    delta=impsig-mu
    eps_del=np.sqrt(delta[:,None, None,None]**2+disp**2)
    phase=kx*a #since Delta=(1,0,0)#
    halfknum = int(knum / 2)
    kind_list=np.arange(halfknum)
    fulllist_P=np.concatenate((kind_list[::-1],kind_list))
    fullindx,fullindy,fullindz=np.meshgrid(fulllist_P,fulllist_P,fulllist_P,indexing='ij')
    allsig=sigmapp[:,fullindx,fullindy,fullindz]
    sigma11=allsig.imag+allsig.real*delta[:,None, None,None]/eps_del
    sigma22=allsig.imag-allsig.real*delta[:,None, None,None]/eps_del
    sigma12=allsig.real*disp/eps_del#*np.exp(1j*phase)
    sigma21=allsig.real*disp/eps_del#*np.exp(-1j*phase)
    return sigma11,sigma22,sigma12,sigma21 # here we give sigmas in full kspace.

def sum_nonlocal_diagrams(sig,knum,a=1):# sum over all nonlocal Sigma_ij(iom)
    k1=np.linspace(-np.pi/a,np.pi/a,num=knum+1)
    k2=np.roll(k1,1)
    kave=(k1+k2)/2
    klist=kave[-knum:] 
    print('klist=',klist)
    k1, k2, k3 = np.meshgrid(klist, klist, klist, indexing='ij')
    kx=0.5*(-k1+k2+k3)
    ky=0.5*(k1-k2+k3)
    kz=0.5*(k1+k2-k3)


    n=np.shape(sig)[0]
    halfknum = int(knum / 2)
    kind_list=np.arange(halfknum)

    # print(np.concatenate((kind_list[::-1],kind_list)))
    # for rotated sigma12 we already have the full k space.
    # fulllist_P=np.concatenate((kind_list[::-1],kind_list))
    # fullindx,fullindy,fullindz=np.meshgrid(fulllist_P,fulllist_P,fulllist_P,indexing='ij')
    # allsig=sig[:,fullindx,fullindy,fullindz]

    # print(allsig[0,3,4,2],allsig[0,6,5,7])
    nonlocal_diagrams=np.zeros(n,dtype=complex)
    Rlist=np.array([-1,0,1])*a
    for Rx in Rlist:
        for Ry in Rlist:
            for Rz in Rlist:
                # if Rx!=0 or Ry!=0 or Rz!=0:
                if Rx*Ry*Rz==0 and np.abs(Rx+Ry+Rz)==1:# 1st NN test
                # if Rx==1 and Ry==1 and Rz==0: # 2nd NN test
                # if Rx==1 and Ry==1 and Rz==1: # 3rd NN test 
                    # for kxind in np.arange(knum):
                    #     for kyind in np.arange(knum):
                    #         for kzind in np.arange(knum):
                    #             for i in np.arange(n):
                    #                 nonlocal_diagrams[i]+=np.exp(1j*(klist[kxind]*Rx+klist[kyind]*Ry+klist[kzind]*Rz))*allsig[i,kxind,kyind,kzind]


                    factor=np.exp(-1j*(kx*Rx+ky*Ry+kz*Rz))
                    for i in np.arange(n):
                        nonlocal_diagrams[i]+=np.sum(sig[i]*factor)
                        # inverse_fourier = np.fft.ifftn(allsig[i])
                        # nonlocal_diagrams[i]+=inverse_fourier[Rx,Ry,Rz]
    return nonlocal_diagrams/knum**3



#----------test functions---------
# all tests are made in the trivial dispersion e_k=0.


def G_test(a=1):
    start_time = time.time()
    U=2.0
    mu=U/2
    T=0.01
    beta=1/T
    sigma=np.loadtxt('{}_{}.dat'.format(U,T))[:500,:]
    sigA=sigma[:,1]+1j*sigma[:,2]
    # sigB=sigma[:,3]+1j*sigma[:,4]
    # z_A=z(beta,mu,sigA)
    # z_B=z(beta,mu,sigB)
    n=sigA.size
    allsigA=ext_sig(beta,sigA)
    # print('n=',n,)
    knum=10
    G_pp,G_mm=G_diagonalized(knum,allsigA,beta,mu)
    # G_A=G_diag_A(knum,z_A,z_B)
    # G_B=G_diag_B(knum,z_A,z_B)
    # G_off=G_offdiag(knum,z_A,z_B)
    fermion_om = (2*np.arange(4*n)+1-4*n)*np.pi/beta
    fermion_iom=1j*fermion_om
    time_G=time.time()
    print("time to calculate prepare 2 G is {:.6f} s".format(time_G-start_time))
    kxind=3
    kyind=4
    kzind=5
    # diagonalized G test

    plt.plot(fermion_om,G_pp[:,kxind,kyind,kzind].real,label='Gk_pp_real')
    plt.plot(fermion_om,G_pp[:,kxind,kyind,kzind].imag,label='Gk_pp_imag')

    plt.plot(fermion_om,G_mm[:,knum-1-kxind,kyind,kzind].real,label='GK-k_mm_real')
    plt.plot(fermion_om,G_mm[:,knum-1-kxind,kyind,kzind].imag,label='GK-k_mm_imag')
    # plt.plot(fermion_om,G_A[:,kxind,kyind,kzind].real,label='Gk_A_real')
    # plt.plot(fermion_om,G_A[:,kxind,kyind,kzind].imag,label='Gk_A_imag')
    # plt.plot(fermion_om,G_A[:,knum-1-kxind,kyind,kzind].real,label='G-k_A_real')
    # plt.plot(fermion_om,G_A[:,knum-1-kxind,kyind,kzind].imag,label='G-k_A_imag')
    # plt.plot(fermion_om,G_B[:,kxind,kyind,kzind].real,label='G_B_real')
    # plt.plot(fermion_om,G_B[:,kxind,kyind,kzind].imag,label='G_B_imag')
    # plt.plot(fermion_om,G_off[:,kxind,kyind,kzind].real,label='Gk_off_real')
    # plt.plot(fermion_om,G_off[:,kxind,kyind,kzind].imag,label='Gk_off_imag')
    # plt.plot(fermion_om,G_off[:,knum-1-kxind,kyind,kzind].real,label='G-k_off_real')
    # plt.plot(fermion_om,G_off[:,knum-1-kxind,kyind,kzind].imag,label='G-k_off_imag')
    # plt.plot(fermion_om,ana_Gk.real,label='ana_real')
    # plt.plot(fermion_om,ana_Gk.imag,label='ana_imag')
    plt.legend()
    plt.show()
    return 0
#clear. k dep clear. sym clear

def precalcP_test(a=1):
    # start_time = time.time()
    
    U=2.0
    mu=U/2
    T=0.01
    beta=1/T
    sigma=np.loadtxt('{}_{}.dat'.format(U,T))[:1000,:]
    sigA=sigma[:,1]+1j*sigma[:,2]
    sigB=sigma[:,3]+1j*sigma[:,4]
    z_A=z(beta,mu,sigA)
    n=sigA.size
    allsigA=ext_sig(beta,sigA)
    allsigB=ext_sig(beta,sigB)
    # print('n=',n,)
    knum=10

    k1=np.linspace(-np.pi/a,np.pi/a,num=knum+1)
    k2=np.roll(k1,1)
    kave=(k1+k2)/2
    klist=kave[-knum:] # this klist is for k points. i.e. not k+q.
    halfknum=int(knum/2)
    # G_11=G_diag_A(knum,z_A,z_B)
    # G_22=G_diag_B(knum,z_A,z_B)
    # G_12=G_offdiag(knum,z_A,z_B)
    G_pp,G_mm=G_diagonalized(knum,allsigA,beta,mu)

      # try to systematically check a random point in P
    qxind=1
    qyind=2
    qzind=3
    Omind=0
    start_time = time.time()

    #old brute-force method
    P11=precalcP_improved(beta,knum,G_pp,allsigA,mu,'up')#
    # P22=precalcP_improved(beta,knum,G_mm,allsigB,mu,'dn')#
    # P11=+precalcP(beta,knum,G_pp,G_pp)
    # P12=+precalcP(beta,knum,G_11,G_12)+precalcP(beta,knum,G_12,G_22)
    # P22=+precalcP_improved(beta,knum,G_22,allsig22,mu)+precalcP(beta,knum,G_12,G_12)
    # new trick
    # P1=precalcP_improved(beta,knum,Gk,allsig,mu)


    # print('numerically,P[{},{},{},{}]='.format(Omind,qxind,qyind,qzind),P11[Omind+n,qxind,qyind,qzind])
    # print('numerically,P[{},{},{},{}]='.format(-Omind,qxind,qyind,qzind),P[-Omind+n,qxind,qyind,qzind])



    end_time = time.time()
    elapsed_time = end_time - start_time
    print("time is {:.6f} s".format(elapsed_time))
    Boson_om = (2*np.arange(2*n+1)-2*n)*np.pi/beta
    Boson_iom=1j*Boson_om
    plt.plot(Boson_om,P11[:,qxind,qyind,qzind].real,label='P11_real')#'BruteForce_real'
    plt.plot(Boson_om,P11[:,qxind,qyind,qzind].imag,label='P11_imag')
    # plt.plot(Boson_om,P22[:,qxind,qyind,qzind].real,label='P22_real')#'BruteForce_real'
    # plt.plot(Boson_om,P22[:,qxind,qyind,qzind].imag,label='P22_imag')
    # plt.plot(Boson_om,P1[:,qxind,qyind,qzind].real,label='trick_real')
    # plt.plot(Boson_om,P1[:,qxind,qyind,qzind].imag,label='trick_imag')
    plt.legend()
    plt.show()
    return 0
#Clear.

def qxfold(qind,knum):#for sigtest 
    if knum % 2 != 0:
        print('knum should be a even number!')
        return -1
    halfknum=int(knum/2)
    if qind>halfknum-1:
        return qind-halfknum
    elif 0<=qind<=halfknum-1:
        return halfknum-1-qind

def sigtest(a=1):
    #take the example: if all G only depend on omega. what happens....
    start_time = time.time()
    U=2.0
    mu=U/2
    T=0.01
    beta=1/T
    sigma=np.loadtxt('{}_{}.dat'.format(U,T))[:500,:]
    sigA=sigma[:,1]+1j*sigma[:,2]
    allsigA=ext_sig(beta,sigA)
    # alp=z(beta,mu,sig,0)
    n=int(allsigA.size/4)
    # print(n)
    knum=10
    G_pp,G_mm=G_diagonalized(knum,allsigA,beta,mu)
    P_pp=precalcP_improved(beta,knum,G_pp,allsigA,mu,'up')
    P_mm=P_pp.conjugate()
    print('shape of P,',np.shape(P_pp))
    # iomlist = 1j*(2*np.arange(2*n)+1-2*n)*np.pi/beta# sum over fermion matsubara freqs
    time_G=time.time()
    print("time to calculate prepare all G and P is {:.6f} s".format(time_G-start_time))

    #--------first, let's check P---------
    # precalcP_test()

    #---------‘analytical’---------
    kxind=3
    kyind=4
    kzind=2
    omind=2
    # k1=np.linspace(-np.pi/a,np.pi/a,num=knum+1)
    # k2=np.roll(k1,1)
    # kave=(k1+k2)/2
    # klist=kave[-knum:] # this klist is for k points.
    # halfknum=int(knum/2)
    # qlist=klist[-halfknum:]

    start_time2 = time.time()
    sig_pp=precalcsig(U,beta,knum,P_mm,G_pp)
    sig_mm=precalcsig(U,beta,knum,P_pp,G_mm)
    sigma11,sigma22,sigma12,sigma21=reverse_diag(knum,sig_pp,allsigA[n:3*n],mu)
    # print('sigma_num[{},{},{},{}]'.format(omind,kxind,kyind,kzind),sig[omind+n,kxind,kyind,kzind])
    time_sig2=time.time()
    print("time to calculate a single numerical sigma is {:.6f} s".format(time_sig2-start_time2))
    plt.plot(sig_pp[:,kxind,kyind,kzind].real,label='sig_pp_real')
    plt.plot(sig_pp[:,kxind,kyind,kzind].imag,label='sig_pp_imag')
    plt.plot(sig_mm[:,kxind,kyind,kzind].real,label='sig_mm_real')
    plt.plot(sig_mm[:,kxind,kyind,kzind].imag,label='sig_mm_imag')
    plt.legend()
    plt.show()
    plt.plot(sigma12[:,kxind,kyind,kzind].real,label='sigma12_real')
    plt.plot(sigma12[:,kxind,kyind,kzind].imag,label='sigma12_imag')
    plt.plot(sigma21[:,kxind,kyind,kzind].real,label='sigma21_real')
    plt.plot(sigma21[:,kxind,kyind,kzind].imag,label='sigma21_imag')
    plt.legend()
    plt.show()
    return 0
#clear. sym clear. ~50s for a sigma.

def nonlocal_diagram(filename,outputname,beta,U,mu,knum=10):
    start_time = time.time()
    n=500
    sigma=np.loadtxt(filename)[:n,:]
    sigA=sigma[:,1]+1j*sigma[:,2]
    sigB=sigma[:,3]+1j*sigma[:,4]

    #a 'symmetrical' trial
    # sigA=mu+0.001+1j*sigma[:,2]
    # sigB=mu-0.001+1j*sigma[:,2]

    if sigma[0,0]/(np.pi/beta)>1.01 or sigma[0,0]/(np.pi/beta)<0.99:
        print('seems the temperature does not match!')
        return 0
    fullsigA=ext_sig(beta,sigA)
    fullsigB=ext_sig(beta,sigB)
    # plt.plot(fullsigA.real,label='unperturbed sigA real')
    # plt.plot(fullsigA.imag,label='unperturbed sigA imag')
    # plt.plot(fullsigB.real,label='unperturbed sigB real')
    # plt.plot(fullsigB.imag,label='unperturbed sigB imag')
    # plt.legend()
    # plt.show()

    G_pp,G_mm=G_diagonalized(knum,fullsigA,beta,mu)

    # print('shape of alpA,B',np.shape(alpA),np.shape(alpB))
    # print('shape of G_k_A,B',np.shape(G_k_A),np.shape(G_k_B))
    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # heatmap1=ax1.imshow(G_k_A[0,0].real, cmap='hot')
    # ax1.set_title('G_A')
    # fig.colorbar(heatmap1,ax=ax1)
    # heatmap2=ax2.imshow(G_k_B[0,0].real, cmap='hot')
    # ax2.set_title('G_B')
    # fig.colorbar(heatmap2,ax=ax2)
    # plt.show()

    # plt.plot(G_k_A[:,1,0,2].real,label='G_A real')
    # plt.plot(G_k_A[:,1,0,2].imag,label='G_A imag')
    # plt.plot(G_k_B[:,1,0,2].real,label='G_B real')
    # plt.plot(G_k_B[:,1,0,2].imag,label='G_B imag')
    # plt.legend()
    # plt.show()


    # brute-force
    # P_A=precalcP(beta,knum,G_k_A,G_kq_A)
    # P_B=precalcP(beta,knum,G_k_B,G_kq_B)
    # trick
    P_pp=precalcP_improved(beta,knum,G_pp,fullsigA,mu,'up')
    # P_mm=P_pp.conjugate()

    # #plot 2 heatmaps
    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # heatmap1=ax1.imshow(P_A[500,0].real, cmap='hot')
    # ax1.set_title('p_A')
    # fig.colorbar(heatmap1,ax=ax1)
    # heatmap2=ax2.imshow(P_B[500,0].real, cmap='hot')
    # ax2.set_title('p_B')
    # fig.colorbar(heatmap2,ax=ax2)
    # plt.show()
    xind=1
    yind=2
    zind=3
    # plt.plot(P_A[:,xind,yind,zind].real,label='P_A real')
    # plt.plot(P_A[:,xind,yind,zind].imag,label='P_A imag')
    # plt.plot(P_B[:,xind,yind,zind].real,label='P_B real')
    # plt.plot(P_B[:,xind,yind,zind].imag,label='P_B imag')
    # plt.legend()
    # plt.show()

    Sigma_mm=precalcsig(U,beta,knum,P_pp,G_mm)
    Sigma_pp=-Sigma_mm.conjugate()
    #put it back to original basis.
    sigma11,sigma22,sigma12,sigma21=reverse_diag(knum,Sigma_pp,fullsigA[n:3*n],mu)

    # Sigma_B=-1*Sigma_A.real+Sigma_A.imag

    # #plot 2 heatmaps
    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # heatmap1=ax1.imshow(Sigma_A[500,0].real, cmap='hot')
    # ax1.set_title('sigma_A[500,0]')
    # fig.colorbar(heatmap1,ax=ax1)
    # heatmap2=ax2.imshow(-Sigma_B[500,4].real, cmap='hot')
    # ax2.set_title('-sigma_B[500,4]')
    # fig.colorbar(heatmap2,ax=ax2)
    # plt.show()

    plt.plot(sigma12[:,0,1,2].real,label='sigma12[:,0,1,2] real')
    plt.plot(sigma12[:,0,1,2].imag,label='sigma12[:,0,1,2]  imag')
    plt.plot(sigma21[:,4,3,2].real,label='sigma21[:,4,3,2] real')
    plt.plot(sigma21[:,4,3,2].imag,label='sigma21[:,4,3,2] imag')
    plt.legend()
    plt.show()

    time_Sig=time.time()
    print("time to calculate all G and P and Sigma is {:.6f} s".format(time_Sig-start_time))
    nonlocaldiagrams_12=sum_nonlocal_diagrams(sigma12,knum)
    # nonlocaldiagrams_mm=sum_nonlocal_diagrams(Sigma_mm,knum)
    # print(nonlocaldiagrams_A[0],nonlocaldiagrams_B[0])
    time_final=time.time()
    print("time to prepare nonlocal diagrams is {:.6f} s".format(time_final-time_Sig))
    f = open(outputname, 'w')
    for i in range(np.shape(sigma)[0]):# consider the case that we may have many columns of Gf
        print(sigma[i,0],end='\t', file=f)# print in the same line
        print(sigA[i].real+nonlocaldiagrams_12[i].real,end='\t', file=f)# print in the same line
        print(sigA[i].imag+nonlocaldiagrams_12[i].imag,end='\t', file=f)# print in the same line
        # print(sigB[i].real+nonlocaldiagrams_mm[i].real,end='\t', file=f)# print in the same line
        # print(sigB[i].imag+nonlocaldiagrams_mm[i].imag,end='\t', file=f)# print in the same line
        print('', file=f)# switch to another line
    f.close()
    plt.plot(fullsigA[n:3*n].real,label='unperturbed sigA real')
    plt.plot(fullsigA[n:3*n].imag,label='unperturbed sigA imag')
    plt.plot(fullsigB[n:3*n].real,label='unperturbed sigB real')
    plt.plot(fullsigB[n:3*n].imag,label='unperturbed sigB imag')
    plt.plot(fullsigA[n:3*n].real+nonlocaldiagrams_12.real,label='corrected sig_pp real')
    plt.plot(fullsigA[n:3*n].imag+nonlocaldiagrams_12.imag,label='corrected sig_pp imag')
    # plt.plot(fullsigB[n:3*n].real+nonlocaldiagrams_mm.real,label='corrected sig_mm real')
    # plt.plot(fullsigB[n:3*n].imag+nonlocaldiagrams_mm.imag,label='corrected sig_mm imag')
    plt.legend()
    plt.show()
    plt.plot(nonlocaldiagrams_12.real,label='sig_pp real correction')
    plt.plot(nonlocaldiagrams_12.imag,label='sig_pp imag correction')
    # plt.plot(nonlocaldiagrams_mm.real,label='sigB real correction')
    plt.legend()
    plt.show()
    return 0

# G_test()
# precalcP_test()
# sigtest()


T=0.01
Uc=2.0
if (len(sys.argv)!=3):
    print('usually we need 2 parameters:T and U.')
    
if (len(sys.argv)==3):
    Uc=float(sys.argv[1])
    T=float(sys.argv[2])
    print('T=',T)
    print('Uc=',Uc)
# nonlocal_diagram('Sig_test_para.dat',2.0,10.0,5.0)
# nonlocal_diagram('Sig_test_afm.dat',20.0,4.0,2.0)
nonlocal_diagram('{}_{}.dat'.format(Uc,T),'Sig.pert_{}_{}.dat'.format(Uc,T),1/T,Uc,Uc/2)
# nonlocal_diagram('Sig.OCA','Sig.pert.dat',1/T,Uc,Uc/2)
