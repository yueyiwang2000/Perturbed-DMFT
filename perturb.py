import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess
import time
from joblib import Parallel, delayed
import perturb_imp


def dispersion(kx,ky,kz,a=1,t=1):
    e_k=-2*t*np.cos(kx*a)-2*t*np.cos(ky*a)-2*t*np.cos(kz*a)
    return e_k

def calc_eps2_ave(knum,a=1):
    kall=np.linspace(-np.pi/a,np.pi/a,num=knum+1)
    kroll=np.roll(kall,1)
    kave=(kall+kroll)/2
    klist=kave[-knum:] 
    k1, k2, k3 = np.meshgrid(klist, klist, klist, indexing='ij')
    kx=0.5*(-k1+k2+k3)
    ky=0.5*(k1-k2+k3)
    kz=0.5*(k1+k2-k3)
    eps2=dispersion(kx, ky, kz)**2
    eps2_ave=np.sum(eps2)/knum**3
    return eps2_ave

def calc_disp(knum,a=1):
    kall=np.linspace(-np.pi/a,np.pi/a,num=knum+1)
    kroll=np.roll(kall,1)
    kave=(kall+kroll)/2
    klist=kave[-knum:] 
    k1, k2, k3 = np.meshgrid(klist, klist, klist, indexing='ij')
    kx=0.5*(-k1+k2+k3)
    ky=0.5*(k1-k2+k3)
    kz=0.5*(k1+k2-k3)
    disp=dispersion(kx, ky, kz)
    return disp

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

def z(beta,mu,sig):
    # sometimes we want values of G beyond the range of n matsubara points. try to do a simple estimation for even higher freqs:
    n=sig.size
    om=(2*np.arange(4*n)+1-4*n)*np.pi/beta
    allsig=ext_sig(beta,sig)
    z=om*1j+mu-allsig
    return z

def ext_sig(beta,sig):
    lenom=sig.size
    # print(lenom)
    all_om=(2*np.arange(2*lenom)+1)*np.pi/beta
    allsig=np.zeros(4*lenom,dtype=complex)
    allsig[2*lenom:3*lenom]=sig
    allsig[3*lenom:4*lenom]=sig[lenom-1].real+1j*sig[lenom-1].imag*all_om[lenom-1]/all_om[lenom:2*lenom]
    allsig[:2*lenom]=allsig[4*lenom:2*lenom-1:-1].conjugate()
    return allsig

def G_11(knum,z_A,z_B,a=1):# and, G_22=-G_diag_11.conj
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


# G12 is real. We are gonna to define it as a real array to accelerate the calculation.
def G_12(knum,z_A,z_B,a=1):
    kall=np.linspace(-np.pi/a,np.pi/a,num=knum+1)
    kroll=np.roll(kall,1)
    kave=(kall+kroll)/2
    klist=kave[-knum:] 
    n=z_A.size
    k1, k2, k3 = np.meshgrid(klist, klist, klist, indexing='ij')
    kx=0.5*(-k1+k2+k3)
    ky=0.5*(k1-k2+k3)
    kz=0.5*(k1+k2-k3)
    G_offdiag=np.zeros((n,knum,knum,knum))
    zazb=z_A*z_B
    dis=dispersion(kx, ky, kz)
    G_offdiag = dis / (zazb[:, None, None, None].real - dis**2)
    return G_offdiag



#here, I specify Omega_index is index of boson freq, as it should be.
# and, let's set qx as sth like [1/10,3/10,5/10,7/10,9/10]. half as the k_f
# this means, qx[0]=1/10. This is an eg in knum=10.


#this precalc P is a brute force version of polarization. Which is for off-diag elements of P.
# And, we only need G12G21=G12G12 like terms.
# Also, since G12 G21 P21 is real, we only define P as real function.
def precalcP_innerloop(q, kxind, kyind, kzind, knum, n, G1):
    qx, qy, qz = q
    P_partial = np.zeros((2*n+1, 1, 1, 1))
    G1_kq = G1[:, foldback(kxind + qx, knum), foldback(kyind + qy, knum), foldback(kzind + qz, knum)]
    G1_sliced = G1[n:3*n]
    for Omind in np.arange(n+1)+n:
        P_partial[Omind, 0, 0, 0] = np.sum(G1_sliced * G1_kq[n+Omind-n:3*n+Omind-n]) 
        P_partial[2*n-Omind, 0, 0, 0]=P_partial[Omind, 0, 0, 0]   
    return P_partial

def precalcP(beta, knum, G1, a=1):
    n = int(np.shape(G1)[0] / 4)
    kind_list = np.arange(knum)
    if knum % 2 != 0:
        print('knum should be a even number!')
        return 0
    halfknum = int(knum / 2)
    qind_list = np.arange(halfknum+1)
    P = np.zeros((2*n+1, halfknum+1, halfknum+1, halfknum+1))
    kxind, kyind, kzind = np.meshgrid(kind_list, kind_list, kind_list, indexing='ij')

    # Flatten the qind_list
    # q_indices = [(qx, qy, qz) for qx in qind_list for qy in qind_list for qz in qind_list]
    q_indices=gen_qindices(qind_list)
    # Parallelize the inner loop
    results = Parallel(n_jobs=-1)(delayed(precalcP_innerloop)(q, kxind, kyind, kzind, knum, n, G1) for q in q_indices)
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

# This is for diagonal elements for polarization. Here the only way to solve the convergence issue
# is that to contour intregration in brute force, which will result in 4 terms. 
# here alpha_k=sqrt(delta_inf**2+eps_k**2)
def precalcP_innerloop_improved(q, kxind, kyind, kzind, knum, n, G_k,G_k0,alphak,f_alphak,deltainf,beta):
    qx, qy, qz = q
    P_partial = np.zeros((2*n+1, 1, 1, 1), dtype=complex)
    G_kq = G_k[:, foldback(kxind + qx, knum), foldback(kyind + qy, knum), foldback(kzind + qz, knum)]
    G_kq0 = G_k0[:, foldback(kxind + qx, knum), foldback(kyind + qy, knum), foldback(kzind + qz, knum)]
    alpha_kq=alphak[foldback(kxind + qx, knum), foldback(kyind + qy, knum), foldback(kzind + qz, knum)]
    falpha_kq=f_alphak[foldback(kxind + qx, knum), foldback(kyind + qy, knum), foldback(kzind + qz, knum)]
    
    for Omind in np.arange(n+1)+n:
        #complex trick
        lindhard1=0.5*(1+deltainf/alphak)*0.5*(1+deltainf/alpha_kq) * (f_alphak-falpha_kq)/(1j*(Omind-n-0.001)*2*np.pi/beta-alpha_kq+alphak)
        lindhard2=0.5*(1-deltainf/alphak)*0.5*(1-deltainf/alpha_kq) * (-f_alphak+falpha_kq)/(1j*(Omind-n-0.001)*2*np.pi/beta+alpha_kq-alphak)
        lindhard3=0.5*(1+deltainf/alphak)*0.5*(1-deltainf/alpha_kq) * (f_alphak+falpha_kq-1)/(1j*(Omind-n-0.001)*2*np.pi/beta+alpha_kq+alphak)
        lindhard4=0.5*(1-deltainf/alphak)*0.5*(1+deltainf/alpha_kq) * (1-f_alphak-falpha_kq)/(1j*(Omind-n-0.001)*2*np.pi/beta-alpha_kq-alphak)
        lindhard=lindhard1+lindhard2+lindhard3+lindhard4

        
        P_partial[Omind, 0, 0, 0] =  +beta*np.sum(lindhard)+np.sum(G_k[n:3*n] * G_kq[n+Omind-n:3*n+Omind-n]-G_k0[n:3*n] * G_kq0[n+Omind-n:3*n+Omind-n])
        
        
        #brute-force 
        # P_partial[Omind, 0, 0, 0]=np.sum(G_k[n:3*n] * G_kq[n+Omind-n:3*n+Omind-n])


        P_partial[2*n-Omind, 0, 0, 0]=P_partial[Omind, 0, 0, 0].conjugate()
    return P_partial.real

def precalcP_improved(beta, knum, Gk, fullsig,mu,a=1):# here, we need original sigma.
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
    delta_inf=np.abs(-mu+fullsig[-1].real)
    # alphak=dispersion(kx,ky,kz)# another alpha for test.
    alphak=np.sqrt(dispersion(kx,ky,kz)**2+delta_inf**2)
    f_alphak=fermi(alphak,beta)
    #generate unperturbed Green's function

    # z_0=z0(beta,mu,fullsig)
    om=(2*np.arange(4*n)+1-4*n)*np.pi/beta
    z_bar=1j*(om)# z_bar is imaginary.-fullsig.imag
    Gk0=np.zeros((n,knum,knum,knum),dtype=np.complex128)
    #G(0)=(i*omega+delta_inf)/((i*omega)**2-alpha**2)
    Gk0 = (z_bar[:, None, None, None]+delta_inf) /(z_bar[:, None, None, None]**2 - alphak**2)

    P = np.zeros((2*n+1, halfknum+1, halfknum+1, halfknum+1))
    # print('shape of P,',np.shape(P))
    # fermion_Omega_ind = np.arange(n)
    kxind, kyind, kzind = np.meshgrid(kind_list, kind_list, kind_list, indexing='ij')

    # Flatten the qind_list
    q_indices=gen_qindices(qind_list)
    # Parallelize the inner loop
    results = Parallel(n_jobs=-1)(delayed(precalcP_innerloop_improved)(q, kxind, kyind, kzind, knum, n, Gk,Gk0,alphak,f_alphak,delta_inf,beta) for q in q_indices)
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
    for omind in np.arange(n):
        sig_partial[omind, 0, 0, 0] = np.sum(P_k * G_kq[omind:omind +2*n+1])# from omind, omind+1, ..., to omind+2n
        sig_partial[2*n-1-omind, 0, 0, 0]=sig_partial[omind, 0, 0, 0].conjugate()
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
    sigB=sigma[:,3]+1j*sigma[:,4]
    z_A=z(beta,mu,sigA)
    z_B=z(beta,mu,sigB)
    n=sigA.size
    knum=10
    G11=G_11(knum,z_A,z_B)
    G22=-G11.conjugate()
    G12=G_12(knum,z_A,z_B)
    fermion_om = (2*np.arange(4*n)+1-4*n)*np.pi/beta
    time_G=time.time()
    print("time to calculate prepare 2 G is {:.6f} s".format(time_G-start_time))
    for kxind in np.arange(10):
        for kyind in np.arange(10):
            for kzind in np.arange(10):
    # kxind=0
    # kyind=0
    # kzind=0

                plt.plot(fermion_om[2*n:3*n],G11[2*n:3*n,kxind,kyind,kzind].real,label='G11_k_real')
                plt.plot(fermion_om[2*n:3*n],G11[2*n:3*n,kxind,kyind,kzind].imag,label='G11_k_imag')
                # plt.plot(fermion_om,G_A[:,knum-1-kxind,kyind,kzind].real,label='G-k_A_real')
                # plt.plot(fermion_om,G_A[:,knum-1-kxind,kyind,kzind].imag,label='G-k_A_imag')
                plt.plot(fermion_om[2*n:3*n],G22[2*n:3*n,kxind,kyind,kzind].real,label='G22_k_real')
                plt.plot(fermion_om[2*n:3*n],G22[2*n:3*n,kxind,kyind,kzind].imag,label='G22_k_imag')
                # plt.plot(fermion_om,G12[:,kxind,kyind,kzind].real,label='G12_k_real')
                # plt.plot(fermion_om,G12[:,kxind,kyind,kzind].imag,label='G12_k_imag')
                plt.legend()
                plt.grid()
                plt.show()
    return 0
#clear

def precalcP_test(a=1):
    U=2.0
    mu=U/2
    T=0.01
    beta=1/T
    sigma=np.loadtxt('{}_{}.dat'.format(U,T))[:500,:]
    sigA=sigma[:,1]+1j*sigma[:,2]
    sigB=sigma[:,3]+1j*sigma[:,4]
    z_A=z(beta,mu,sigA)
    z_B=z(beta,mu,sigB)
    n=sigA.size
    allsigA=ext_sig(beta,sigA)
    allsigB=ext_sig(beta,sigB)
    # print('n=',n,)
    knum=10

    G11=G_11(knum,z_A,z_B)
    # G22=-G11.conjugate()
    G12=G_12(knum,z_A,z_B)
    # G_pp,G_mm=G_diagonalized(knum,allsigA,beta,mu)

      # try to systematically check a random point in P
    qxind=0
    qyind=0
    qzind=0
    Omind=0


    start_time = time.time()
    #brute-force method: for P12=P21
    P12=precalcP(beta,knum,G12)

    # new trick
    P11=precalcP_improved(beta,knum,G11,allsigA,mu)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("time is {:.6f} s".format(elapsed_time))
    Boson_om = (2*np.arange(2*n+1)-2*n)*np.pi/beta
    # Boson_iom=1j*Boson_om
    plt.plot(Boson_om,P11[:,qxind,qyind,qzind].real,label='P11_real')
    # plt.plot(Boson_om,P11[:,qxind,qyind,qzind].imag,label='P11_imag')
    # plt.plot(Boson_om,P22[:,qxind,qyind,qzind].real,label='P22_real')
    # plt.plot(Boson_om,P22[:,qxind,qyind,qzind].imag,label='P22_imag')
    plt.plot(Boson_om,P12[:,qxind,qyind,qzind].real,label='P12_real')
    plt.plot(Boson_om,P12[:,qxind,qyind,qzind].imag,label='P12_imag')
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


def sig_imp_test(U,T,knum):
    # U=2.0
    mu=U/2
    # T=0.01
    beta=1/T
    sigma=np.loadtxt('{}_{}.dat'.format(U,T))[:500,:]
    sigA=sigma[:,1]+1j*sigma[:,2]
    sigB=sigma[:,3]+1j*sigma[:,4]
    z_A=z(beta,mu,sigA)
    z_B=z(beta,mu,sigB)
    n=sigA.size
    allsigA=ext_sig(beta,sigA)
    allsigB=ext_sig(beta,sigB)
    delta_inf=np.abs(-mu+allsigA[-1].real)
    # knum=10
    eps2_ave=calc_eps2_ave(knum)
    G11=G_11(knum,z_A,z_B)
    G22=-G11.conjugate()
    # print(np.shape(G11),np.shape(G22))
    G11_imp=np.sum(G11,axis=(1,2,3))/knum**3
    G22_imp=np.sum(G22,axis=(1,2,3))/knum**3
    sigimp11,sigimp22=perturb_imp.pertimp_func(G11_imp,G22_imp,delta_inf,beta,U,eps2_ave)
    return sigimp11,sigimp22

def new_sig(U,T,knum,n,a=1):
    start_time = time.time()
    mu=U/2
    beta=1/T
    sigma=np.loadtxt('{}_{}.dat'.format(U,T))[:n,:]
    sigA=sigma[:,1]+1j*sigma[:,2]#sig+delta
    sigB=sigma[:,3]+1j*sigma[:,4]#sig-delta
    z_A=z(beta,mu,sigA)#z-delta
    z_B=z(beta,mu,sigB)#z+delta
    # n=sigA.size
    allsigA=ext_sig(beta,sigA)
    allsigB=ext_sig(beta,sigB)
 
    G11=G_11(knum,z_A,z_B)
    # G22=-G11.conjugate()
    G12=G_12(knum,z_A,z_B)
    time_G=time.time()
    print("time to calculate prepare all G is {:.6f} s".format(time_G-start_time))
    P12=precalcP(beta,knum,G12)
    P11=precalcP_improved(beta,knum,G11,allsigA,mu)
    time_P=time.time()
    print("time to calculate prepare all G and P is {:.6f} s".format(time_P-time_G))

    kxind=0
    kyind=0
    kzind=0
    # omind=2
    sig_11=precalcsig(U,beta,knum,P11,G11)# actually P22 and G11. BUT P11=P22
    sig_22=-sig_11.conjugate()
    sig_new_12=precalcsig(U,beta,knum,P12,G12)
    time_sig=time.time()
    print("time to calculate a single numerical sigma is {:.6f} s".format(time_sig-time_P))
    # plt.plot(sig_11[:,kxind,kyind,kzind].real,label='sig_11_real')
    # plt.plot(sig_11[:,kxind,kyind,kzind].imag,label='sig_11_imag')
    # plt.plot(sig_22[:,kxind,kyind,kzind].real,label='sig_22_real')
    # plt.plot(sig_22[:,kxind,kyind,kzind].imag,label='sig_22_imag')
    # plt.plot(sig_12[:,kxind,kyind,kzind].real,label='sig_12_real')
    # plt.plot(sig_12[:,kxind,kyind,kzind].imag,label='sig_12_imag')
    # plt.legend()
    # plt.show()
    sigimp11,sigimp22=sig_imp_test(U,T,knum)
    # plt.plot(allsigA[n:3*n].real,label='sig_imp_11_real')
    # plt.plot(allsigA[n:3*n].imag,label='sig_imp_11_imag')
    # plt.plot(allsigB[n:3*n].real,label='sig_imp_22_real')
    # plt.plot(allsigB[n:3*n].imag,label='sig_imp_22_imag')
    sig_new_11=sig_11+allsigA[n:3*n, None, None, None]-sigimp11[:, None, None, None]
    sig_new_22=sig_22+allsigB[n:3*n, None, None, None]-sigimp22[:, None, None, None]
    # plt.plot(sig_new_11[:,kxind,kyind,kzind].real,label='sig_new_11_real')
    # plt.plot(sig_new_11[:,kxind,kyind,kzind].imag,label='sig_new_11_imag')
    # plt.plot(sig_new_22[:,kxind,kyind,kzind].real,label='sig_new_22_real')
    # plt.plot(sig_new_22[:,kxind,kyind,kzind].imag,label='sig_new_22_imag')
    # plt.plot(sig_new_12[:,kxind,kyind,kzind].real,label='sig_new_12_real')
    # plt.plot(sig_new_12[:,kxind,kyind,kzind].imag,label='sig_new_12_imag')
    # plt.legend()
    # plt.show()

    return sig_new_11,sig_new_22,sig_new_12
#clear. 

def impurity_test():
    U=2.0
    T=0.01
    mu=U/2
    knum=10
    beta=1/T
    n=500
    halfknum=int(knum/2)

    iom= 1j*(2*np.arange(2*n)+1-2*n)*np.pi/beta
    fermion_om=(2*np.arange(n)+1)*np.pi/beta
    # to generate dispertion 
    
    # halfkind=np.arange(halfknum)+halfknum
    disp=calc_disp(knum)#[halfknum:knum,halfknum:knum,halfknum:knum]
    # print('shape of disp',np.shape(disp))


    # just for test. old sigma.
    sigma=np.loadtxt('{}_{}.dat'.format(U,T))[:500,:]
    sigA=sigma[:,1]+1j*sigma[:,2]
    sigB=sigma[:,3]+1j*sigma[:,4]
    allsigA=ext_sig(beta,sigA)[n:3*n]
    allsigB=ext_sig(beta,sigB)[n:3*n]
    Gk_11=(iom[:, None, None, None]+mu-allsigA[:, None, None, None])/((iom[:, None, None, None]+mu-allsigA[:, None, None, None])*(iom[:, None, None, None]+mu-allsigB[:, None, None, None])-(disp[None, :, :, :])**2)
    Gk_22=(iom[:, None, None, None]+mu-allsigB[:, None, None, None])/((iom[:, None, None, None]+mu-allsigA[:, None, None, None])*(iom[:, None, None, None]+mu-allsigB[:, None, None, None])-(disp[None, :, :, :])**2)
    Gk_imp_11=np.sum(Gk_11,axis=(1,2,3))/knum**3
    Gk_imp_22=np.sum(Gk_22,axis=(1,2,3))/knum**3
    plt.plot(fermion_om,Gk_imp_11[n:2*n].real,label='Gk_imp_11 real')
    plt.plot(fermion_om,Gk_imp_11[n:2*n].imag,label='Gk_imp_11 imag')
    plt.plot(fermion_om,Gk_imp_22[n:2*n].real,label='Gk_imp_22 real')
    plt.plot(fermion_om,Gk_imp_22[n:2*n].imag,label='Gk_imp_22 imag')
    # k1=0
    # k2=0
    # k3=0
    # plt.plot(fermion_om,Gk_11[n:2*n,k1,k2,k3].real,label='Gk_11 real')
    # plt.plot(fermion_om,Gk_11[n:2*n,k1,k2,k3].imag,label='Gk_11 imag')
    # plt.plot(fermion_om,Gk_22[n:2*n,k1,k2,k3].real,label='Gk_22 real')
    # plt.plot(fermion_om,Gk_22[n:2*n,k1,k2,k3].imag,label='Gk_22 imag')
    plt.legend()
    plt.grid()
    plt.show()
    # return 0





    sig_new_11,sig_new_22,sig_new_12=new_sig(U,T,knum,n)
    sig_imp_new_11=np.sum(sig_new_11,axis=(1,2,3))/halfknum**3
    sig_imp_new_22=np.sum(sig_new_22,axis=(1,2,3))/halfknum**3

    #output test
    # fnewsig='test_new_sig.imp'
    # f = open(fnewsig, 'w')
    # for i,iom in enumerate(fermion_om):
    #     print(iom, sig_imp_new_11[i].real, sig_imp_new_11[i].imag, sig_imp_new_22[i].real, sig_imp_new_22[i].imag, file=f) 
    # f.close()
    plt.plot(fermion_om,sig_imp_new_11[n:2*n].real,label='sig_imp_new_11 real')
    plt.plot(fermion_om,sig_imp_new_11[n:2*n].imag,label='sig_imp_new_11 imag')
    plt.plot(fermion_om,sig_imp_new_22[n:2*n].real,label='sig_imp_new_22 real')
    plt.plot(fermion_om,sig_imp_new_22[n:2*n].imag,label='sig_imp_new_22 imag')
    plt.legend()
    plt.grid()
    plt.show()
    Gk_new_11=(iom[:, None, None, None]+mu-sig_new_11)/((iom[:, None, None, None]+mu-sig_new_11)*(iom[:, None, None, None]+mu-sig_new_22)-(disp[None, :, :, :]+sig_new_12)**2)#
    Gk_new_22=(iom[:, None, None, None]+mu-sig_new_22)/((iom[:, None, None, None]+mu-sig_new_11)*(iom[:, None, None, None]+mu-sig_new_22)-(disp[None, :, :, :]+sig_new_12)**2)#

    # plt.plot(fermion_om,Gk_new_11[n:2*n,1,2,3].real,label='Gk_imp_new_11[1,2,3] real')
    # plt.plot(fermion_om,Gk_new_11[n:2*n,1,2,3].imag,label='Gk_imp_new_11[1,2,3] imag')
    # plt.plot(fermion_om,Gk_new_22[n:2*n,1,2,3].real,label='Gk_imp_new_22[1,2,3] real')
    # plt.plot(fermion_om,Gk_new_22[n:2*n,1,2,3].imag,label='Gk_imp_new_22[1,2,3] imag')
    plt.legend()
    plt.grid()
    plt.show()
    
    Gk_imp_new_11=np.sum(Gk_new_11,axis=(1,2,3))/halfknum**3
    Gk_imp_new_22=np.sum(Gk_new_22,axis=(1,2,3))/halfknum**3
    plt.plot(fermion_om,Gk_imp_new_11[n:2*n].real,label='Gk_imp_new_11 real')
    plt.plot(fermion_om,Gk_imp_new_11[n:2*n].imag,label='Gk_imp_new_11 imag')
    plt.plot(fermion_om,Gk_imp_new_22[n:2*n].real,label='Gk_imp_new_22 real')
    plt.plot(fermion_om,Gk_imp_new_22[n:2*n].imag,label='Gk_imp_new_22 imag')
    plt.legend()
    plt.grid()
    plt.show()
    Delta_11=iom+mu-sig_imp_new_11-1/Gk_imp_new_11
    Delta_22=iom+mu-sig_imp_new_22-1/Gk_imp_new_22
    plt.plot(fermion_om,Delta_11[n:2*n].real,label='Delta_11 real')
    plt.plot(fermion_om,Delta_11[n:2*n].imag,label='Delta_11 imag')
    plt.plot(fermion_om,Delta_22[n:2*n].real,label='Delta_22 real')
    plt.plot(fermion_om,Delta_22[n:2*n].imag,label='Delta_22 imag')
    plt.legend()
    plt.grid()
    plt.show()
    return Delta_11,Delta_22



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

    # G_pp,G_mm=G_diagonalized(knum,fullsigA,beta,mu)

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
    # P_pp=precalcP_improved(beta,knum,G_pp,fullsigA,mu,'up')
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

    # Sigma_mm=precalcsig(U,beta,knum,P_pp,G_mm)
    # Sigma_pp=-Sigma_mm.conjugate()
    #put it back to original basis.
    # sigma11,sigma22,sigma12,sigma21=reverse_diag(knum,Sigma_pp,fullsigA[n:3*n],mu)

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

    # plt.plot(sigma12[:,0,1,2].real,label='sigma12[:,0,1,2] real')
    # plt.plot(sigma12[:,0,1,2].imag,label='sigma12[:,0,1,2]  imag')
    # plt.plot(sigma21[:,4,3,2].real,label='sigma21[:,4,3,2] real')
    # plt.plot(sigma21[:,4,3,2].imag,label='sigma21[:,4,3,2] imag')
    plt.legend()
    plt.show()

    time_Sig=time.time()
    print("time to calculate all G and P and Sigma is {:.6f} s".format(time_Sig-start_time))
    # nonlocaldiagrams_12=sum_nonlocal_diagrams(sigma12,knum)
    # nonlocaldiagrams_mm=sum_nonlocal_diagrams(Sigma_mm,knum)
    # print(nonlocaldiagrams_A[0],nonlocaldiagrams_B[0])
    time_final=time.time()
    print("time to prepare nonlocal diagrams is {:.6f} s".format(time_final-time_Sig))
    f = open(outputname, 'w')
    for i in range(np.shape(sigma)[0]):# consider the case that we may have many columns of Gf
        print(sigma[i,0],end='\t', file=f)# print in the same line
        # print(sigA[i].real+nonlocaldiagrams_12[i].real,end='\t', file=f)# print in the same line
        # print(sigA[i].imag+nonlocaldiagrams_12[i].imag,end='\t', file=f)# print in the same line
        # print(sigB[i].real+nonlocaldiagrams_mm[i].real,end='\t', file=f)# print in the same line
        # print(sigB[i].imag+nonlocaldiagrams_mm[i].imag,end='\t', file=f)# print in the same line
        print('', file=f)# switch to another line
    f.close()
    plt.plot(fullsigA[n:3*n].real,label='unperturbed sigA real')
    plt.plot(fullsigA[n:3*n].imag,label='unperturbed sigA imag')
    plt.plot(fullsigB[n:3*n].real,label='unperturbed sigB real')
    plt.plot(fullsigB[n:3*n].imag,label='unperturbed sigB imag')
    # plt.plot(fullsigA[n:3*n].real+nonlocaldiagrams_12.real,label='corrected sig_pp real')
    # plt.plot(fullsigA[n:3*n].imag+nonlocaldiagrams_12.imag,label='corrected sig_pp imag')
    # plt.plot(fullsigB[n:3*n].real+nonlocaldiagrams_mm.real,label='corrected sig_mm real')
    # plt.plot(fullsigB[n:3*n].imag+nonlocaldiagrams_mm.imag,label='corrected sig_mm imag')
    plt.legend()
    plt.show()
    # plt.plot(nonlocaldiagrams_12.real,label='sig_pp real correction')
    # plt.plot(nonlocaldiagrams_12.imag,label='sig_pp imag correction')
    # plt.plot(nonlocaldiagrams_mm.real,label='sigB real correction')
    plt.legend()
    plt.show()
    return 0

# G_test()
# precalcP_test()
# sigtest()
impurity_test()


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
# nonlocal_diagram('{}_{}.dat'.format(Uc,T),'Sig.pert_{}_{}.dat'.format(Uc,T),1/T,Uc,Uc/2)
# nonlocal_diagram('Sig.OCA','Sig.pert.dat',1/T,Uc,Uc/2)