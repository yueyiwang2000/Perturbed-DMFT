import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess
import time
from joblib import Parallel, delayed
import perturb_imp
from multiprocessing import Pool

cores_used=6


def dispersion(k1,k2,k3,a=1,t=1):
    kx=(-k1+k2+k3)*np.pi/a
    ky=(k1-k2+k3)*np.pi/a
    kz=(k1+k2-k3)*np.pi/a
    e_k=-2*t*np.cos(kx*a)-2*t*np.cos(ky*a)-2*t*np.cos(kz*a)
    return e_k

def calc_eps2_ave(knum,a=1):
    k1, k2, k3 = gen_full_kgrids(knum)
    eps2=dispersion(k1, k2, k3)**2
    eps2_ave=np.sum(eps2)/knum**3
    return eps2_ave

def calc_disp(knum,a=1):
    k1, k2, k3 = gen_full_kgrids(knum)
    disp=dispersion(k1, k2, k3)
    return disp

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

def gen_full_kgrids(knum,a=1):
    kall=np.linspace(0,1,num=knum+1)
    # kroll=np.roll(kall,1)
    # kave=(kall+kroll)/2
    klist=kall[:knum] 
    # print('klist=',klist)
    k1, k2, k3 = np.meshgrid(klist, klist, klist, indexing='ij')
    # kx=0.5*(-k1+k2+k3)
    # ky=0.5*(k1-k2+k3)
    # kz=0.5*(k1+k2-k3)
    return k1,k2,k3

def calc_sym_point(in_k123,mat,knum):
    S=np.array([[-1,1,1],
                [1,-1,1],
                [1,1,-1]],dtype=int)#(nx,ny,nz)^T=S@(n1,n2,n3)^T
    S_inv=np.linalg.inv(S)
    out_kpoint=(S_inv@mat@S@in_k123).astype(int)
    out_kpoint_mod=np.mod(out_kpoint,knum)
    sgn=(-1)**(np.sum(out_kpoint_mod-out_kpoint)/knum)
    output=np.vstack((out_kpoint_mod,np.array([sgn])))
    # print(output)
    # return out__.astype(int)
    return output.astype(int)


def sym_mapping(k1ind,k2ind,k3ind,knum=10):
    # generally, both G, P, sigma has the same k space symmetry.
    # But hence we have a BCC instead of simple cubic, we have to use linear algebra to figure out the symmetry.
    # define symmetry matrices in x,y,z basis:
    # x_inverse=np.diag([-1,1,1])#x->-x
    # y_inverse=np.diag([1,-1,1])#y->-y
    # z_inverse=np.diag([1,1,-1])#z->-z
    xy_swap=np.array([[0,1,0],
                      [1,0,0],
                      [0,0,1]],dtype=int)# (kx,ky,kz)=(ky,kx,kz), and so on.
    xz_swap=np.array([[0,0,1],
                      [0,1,0],
                      [1,0,0]],dtype=int)
    yz_swap=np.array([[1,0,0],
                      [0,0,1],
                      [0,1,0]],dtype=int)
    # S=np.array([[-1,1,1],
    #             [1,-1,1],
    #             [1,1,-1]],dtype=int)#(nx,ny,nz)^T=S@(n1,n2,n3)^T
    # S_inv=np.linalg.inv(S)
    input_k123=np.array([[k1ind],[k2ind],[k3ind]])
    sym_points=np.array([[k1ind],[k2ind],[k3ind],[1]])# the last element means the sign.
    # ------test code------
    # print('input_k123\n',input_k123)
    # input_kxyz=S@input_k123
    # print('input_kxyz, unit:pi/knum/a\n',input_kxyz)
    # output_kxyz=x_inverse@input_kxyz
    # print('output_kxyz, unit:pi/knum/a\n',output_kxyz)
    # output_k=(S_inv@output_kxyz).astype(int)
    # shifted_output_k=np.mod(output_k,knum)
    # shifted_output_k=np.mod((S_inv@x_inverse@S@input_k123).astype(int),knum)
    # ------end of test code------
    # I think I have another 47 elements in this group.... I have to list all of them.
    # try to apply the generator in the group.
    for xinv in [-1,1]:
        for yinv in [-1,1]:
            for zinv in [-1,1]:
                inv=np.diag([xinv,yinv,zinv])
                # print(xinv,yinv,zinv)
                sym_points=np.concatenate([sym_points,calc_sym_point(input_k123,inv,knum)],axis=1)#xyz
                sym_points=np.concatenate([sym_points,calc_sym_point(input_k123,inv@xy_swap,knum)],axis=1)#yxz
                sym_points=np.concatenate([sym_points,calc_sym_point(input_k123,inv@xz_swap,knum)],axis=1)#zyx 
                sym_points=np.concatenate([sym_points,calc_sym_point(input_k123,inv@yz_swap,knum)],axis=1)#xzy
                # sym_points=np.concatenate([sym_points,calc_sym_point(input_k123,xy_swap@inv,knum)],axis=1)
                # sym_points=np.concatenate([sym_points,calc_sym_point(input_k123,xz_swap@inv,knum)],axis=1)
                # sym_points=np.concatenate([sym_points,calc_sym_point(input_k123,yz_swap@inv,knum)],axis=1)
                sym_points=np.concatenate([sym_points,calc_sym_point(input_k123,inv@xy_swap@xz_swap,knum)],axis=1)#zxy
                sym_points=np.concatenate([sym_points,calc_sym_point(input_k123,inv@xz_swap@xy_swap,knum)],axis=1)#yzx
    # print('shifted_k123\n',shifted_output_k)
    
    output_sym_points=sym_points.T
    # print('output_sym_points\n',output_sym_points)
    return output_sym_points


def calc_sym_array(knum):
    sym_array=np.zeros((knum,knum,knum))
    sym_index=0
    essential_kpoints=np.empty((0,3))
    for i in np.arange(knum):
        for j in np.arange(knum):
            for k in np.arange(knum):
                if sym_array[i,j,k]==0:
                    new_kpoint=np.array([[i,j,k]])
                    # np.concatenate((essential_kpoints,[i,j,k]),axis=0)
                    essential_kpoints=np.vstack((essential_kpoints,new_kpoint))
                    sym_index=sym_index+1
                    sym_points=sym_mapping(i,j,k,knum)
                    for point in sym_points:
                        sym_array[point[0],point[1],point[2]]=sym_index*point[3]
    # print('max_sym_index',sym_index)
    # print('essential_kpoints\n',essential_kpoints.astype(int))
    # print(sym_array)
    return sym_index,essential_kpoints.astype(int),sym_array


def G_11(knum,z_A,z_B,a=1):# and, G_22=-G_diag_11.conj
    n=z_A.size
    k1,k2,k3=gen_full_kgrids(knum,a)
    G_diag_A=np.zeros((n,knum,knum,knum),dtype=np.complex128)
    zazb=z_A*z_B
    G_diag_A = z_B[:, None, None, None] / (zazb[:, None, None, None] - dispersion(k1,k2,k3)**2)
    return G_diag_A

# G12 is real, or effectively can be treated as real. We are gonna to define it as a real array to accelerate the calculation.
def G_12(knum,z_A,z_B,a=1):
    k1,k2,k3=gen_full_kgrids(knum,a)
    n=z_A.size
    G_offdiag=np.zeros((n,knum,knum,knum))
    zazb=z_A*z_B
    dis=dispersion(k1, k2, k3)
    G_offdiag = dis / (zazb[:, None, None, None].real - dis**2)
    return G_offdiag


#this precalc P is a brute force version of polarization. Which is ONLY for off-diag elements of P.
# And, we only need G12G21=G12G12 like terms.
# Also, since G12 G21 P21 is real, we only define P as real function.
def precalcP12_innerloop(q, kxind, kyind, kzind, knum, n, G12):
    qx=q[0]
    qy=q[1]
    qz=q[2]
    P_partial = np.zeros((2*n+1, 1, 1, 1))
    G_12_factor=(-1)**((np.mod(kxind + qx, knum)-(kxind+qx))/knum+(np.mod(kyind + qy, knum)-(kyind+qy))/knum+(np.mod(kzind + qz, knum)-(kzind+qz))/knum)
    G12_kq = G_12_factor*G12[:, np.mod(kxind + qx, knum), np.mod(kyind + qy, knum), np.mod(kzind + qz, knum)]
    G12_sliced = G12[:3*n]
    for Omind in np.arange(n+1)+n:
        P_partial[Omind, 0, 0, 0] = np.sum(G12_sliced * G12_kq[Omind-n:3*n+Omind-n]) 
        P_partial[2*n-Omind, 0, 0, 0]=P_partial[Omind, 0, 0, 0]   
    return P_partial


#chetgpt gave me this. This might be fast but requires huge RAM......
# def precalcP12_innerloop_fast(q, kxind, kyind, kzind, knum, n, G12):
#     qx=q[0]
#     qy=q[1]
#     qz=q[2]
#     G_12_factor=(-1)**((np.mod(kxind + qx, knum)-(kxind+qx))/knum+(np.mod(kyind + qy, knum)-(kyind+qy))/knum+(np.mod(kzind + qz, knum)-(kzind+qz))/knum)
#     G12_kq = G_12_factor*G12[:, np.mod(kxind + qx, knum), np.mod(kyind + qy, knum), np.mod(kzind + qz, knum)]
#     G12_sliced = G12[:3*n]

#     G12_sliced = G12_sliced[:, np.newaxis]
#     P_partial = np.sum(G12_sliced * G12_kq, axis=0)
    
#     P_partial = np.expand_dims(P_partial, axis=(1,2,3))
#     P_partial = np.concatenate([P_partial[n+1:], P_partial[:n+1]], axis=0)

#     return P_partial


def precalcP12(beta, knum, G1, a=1):
    n = int(np.shape(G1)[0] / 4)
    kind_list = np.arange(knum)
    if knum % 2 != 0:
        print('knum should be a even number!')
        return 0
    halfknum = int(knum / 2)
    qind_list = np.arange(halfknum+1)
    P = np.zeros((2*n+1, knum, knum, knum))
    kxind, kyind, kzind = np.meshgrid(kind_list, kind_list, kind_list, indexing='ij')
    max_sym_index,essential_kpoints, sym_array=calc_sym_array(knum)
    # Parallelize the inner loop
    results = Parallel(n_jobs=cores_used)(delayed(precalcP12_innerloop)(q, kxind, kyind, kzind, knum, n, G1) for q in essential_kpoints)
    # Combine the results
    for i, q in enumerate(essential_kpoints):
        qx=q[0]
        qy=q[1]
        qz=q[2]
        # sym_value=sym_array[qx,qy,qz]
        res=np.squeeze(results[i])
        all_sym_kpoints=sym_mapping(qx,qy,qz,knum)
        for kpoint in all_sym_kpoints:
            P[:,kpoint[0],kpoint[1],kpoint[2]]=res*kpoint[3]
    return P / beta * (1/ a / knum) ** 3#2 * np.pi 



def fermi(eps,beta):
    return 1/(np.exp(beta*eps)+1)

# This is for diagonal elements for polarization. Here the only way to solve the convergence issue
# is that to contour intregration in brute force, which will result in 4 terms. 
# here alpha_k=sqrt(delta_inf**2+eps_k**2)
def precalcP11_innerloop(q, kxind, kyind, kzind, knum, n, G_k,G_k0,alphak,f_alphak,deltainf,beta):
    qx=q[0]
    qy=q[1]
    qz=q[2]
    P_partial = np.zeros((2*n+1, 1, 1, 1), dtype=complex)
    G_kq = G_k[:, np.mod(kxind + qx, knum), np.mod(kyind + qy, knum), np.mod(kzind + qz, knum)]
    G_kq0 = G_k0[:, np.mod(kxind + qx, knum), np.mod(kyind + qy, knum), np.mod(kzind + qz, knum)]
    alpha_kq=alphak[np.mod(kxind + qx, knum), np.mod(kyind + qy, knum), np.mod(kzind + qz, knum)]
    falpha_kq=f_alphak[np.mod(kxind + qx, knum), np.mod(kyind + qy, knum), np.mod(kzind + qz, knum)]
    Gk_slice=G_k[:3*n]
    Gk0_slice=G_k0[:3*n]
    for Omind in np.arange(n+1)+n:
        #complex trick
        lindhard1=0.5*(1+deltainf/alphak)*0.5*(1+deltainf/alpha_kq) * (f_alphak-falpha_kq)/(1j*(Omind-n-0.001)*2*np.pi/beta-alpha_kq+alphak)
        lindhard2=0.5*(1-deltainf/alphak)*0.5*(1-deltainf/alpha_kq) * (-f_alphak+falpha_kq)/(1j*(Omind-n-0.001)*2*np.pi/beta+alpha_kq-alphak)
        lindhard3=0.5*(1+deltainf/alphak)*0.5*(1-deltainf/alpha_kq) * (f_alphak+falpha_kq-1)/(1j*(Omind-n-0.001)*2*np.pi/beta+alpha_kq+alphak)
        lindhard4=0.5*(1-deltainf/alphak)*0.5*(1+deltainf/alpha_kq) * (1-f_alphak-falpha_kq)/(1j*(Omind-n-0.001)*2*np.pi/beta-alpha_kq-alphak)
        lindhard=lindhard1+lindhard2+lindhard3+lindhard4

        
        P_partial[Omind, 0, 0, 0] =  +beta*np.sum(lindhard)+np.sum(Gk_slice * G_kq[Omind-n:3*n+Omind-n]-Gk0_slice * G_kq0[Omind-n:3*n+Omind-n])
        
        
        #brute-force 
        # P_partial[Omind, 0, 0, 0]=np.sum(G_k[n:3*n] * G_kq[n+Omind-n:3*n+Omind-n])


        P_partial[2*n-Omind, 0, 0, 0]=P_partial[Omind, 0, 0, 0].conjugate()
    return P_partial.real

def precalcP11(beta, knum, Gk, fullsig,mu,a=1):# here, we need original sigma.
    n = int(np.shape(Gk)[0] / 4)
    kind_list = np.arange(knum) 
    if knum % 2 != 0:
        print('knum should be a even number!')
        return 0
    # halfknum = int(knum / 2)
    # qind_list = np.arange(halfknum+1)
    #prepare sth for this trick.
    max_sym_index,essential_kpoints, sym_array=calc_sym_array(knum)
    k1,k2,k3=gen_full_kgrids(knum)

    #generate alpha, f(alpha) for freq sum
    delta_inf=np.abs(-mu+fullsig[-1].real)
    # alphak=dispersion(kx,ky,kz)# another alpha for test.
    alphak=np.sqrt(dispersion(k1,k2,k3)**2+delta_inf**2)
    f_alphak=fermi(alphak,beta)
    #generate unperturbed Green's function

    om=(2*np.arange(4*n)+1-4*n)*np.pi/beta
    z_bar=1j*(om)# z_bar is imaginary.-fullsig.imag
    Gk0=np.zeros((n,knum,knum,knum),dtype=np.complex128)
    Gk0 = (z_bar[:, None, None, None]+delta_inf) /(z_bar[:, None, None, None]**2 - alphak**2)

    P = np.zeros((2*n+1, knum, knum, knum))
    # print('shape of P,',np.shape(P))
    # fermion_Omega_ind = np.arange(n)
    kxind, kyind, kzind = np.meshgrid(kind_list, kind_list, kind_list, indexing='ij')

    results = Parallel(n_jobs=cores_used)(delayed(precalcP11_innerloop)(q, kxind, kyind, kzind, knum, n, Gk,Gk0,alphak,f_alphak,delta_inf,beta) for q in essential_kpoints)
    for i, q in enumerate(essential_kpoints):
        qx=q[0]
        qy=q[1]
        qz=q[2]
        res=np.squeeze(results[i])
        all_sym_kpoints=sym_mapping(qx,qy,qz,knum)
        for kpoint in all_sym_kpoints:
            P[:,kpoint[0],kpoint[1],kpoint[2]]=res
    return P / beta * (1/ a / knum) ** 3

# Note: remember G and P has different spin.
#since every sigma_k can be only used once, seems we don't need to store them in RAM.
# However, for a specific k point, we still want sigma at every fermion matsubara freq.

# as polarization function, I specify omega_index is index of both fremion and boson freq.

def precalcsig_innerloop(k, qxind, qyind, qzind, knum, n, P_k, Gk,opt):
    kx=k[0]
    ky=k[1]
    kz=k[2]
    if opt==12:
        G_12_factor=(-1)**((np.mod(qxind + kx, knum)-(qxind+kx))/knum+(np.mod(qyind + ky, knum)-(qyind+ky))/knum+(np.mod(qzind + kz, knum)-(qzind+kz))/knum)
    elif opt==11:
        G_12_factor=1
    else:
        print('please specify 12 or 11!')
        return 0
    sig_partial = np.zeros((2*n, 1, 1, 1), dtype=complex)
    G_kq = G_12_factor*Gk[:, np.mod(qxind + kx, knum), np.mod(qyind + ky, knum), np.mod(qzind + kz, knum)]
    for omind in np.arange(n):
        sig_partial[omind, 0, 0, 0] = np.sum(P_k * G_kq[omind:omind +2*n+1])# from omind, omind+1, ..., to omind+2n
        sig_partial[2*n-1-omind, 0, 0, 0]=sig_partial[omind, 0, 0, 0].conjugate()
    # print(k,sig_partial[n,0,0,0])
    # plt.plot(sig_partial[n:3*n,0,0,0].real,label='sigk_pert_11_real')
    # plt.plot(sig_partial[n:3*n,0,0,0].imag,label='sigk_pert_11_imag')
    # plt.grid()
    # plt.legend()
    # plt.show()
    return sig_partial

def precalcsig(U,beta, knum, Pk, Gk, opt,a=1):
    n = int(np.shape(Gk)[0]/4)
    # print('n=',n)
    max_sym_index,essential_kpoints, sym_array=calc_sym_array(knum)
    if opt==12:
        power=1
    elif opt==11:
        power=2
    else:
        print('please specify 12 or 11!')
        return 0
    if knum % 2 != 0:
        print('knum should be a even number!')
        return 0
    
    # halfknum = int(knum / 2)
    qind_list = np.arange(knum)
    sig = np.zeros((2*n, knum, knum, knum), dtype=complex)
    qxind, qyind, qzind = np.meshgrid(qind_list, qind_list, qind_list, indexing='ij')

    # Flatten the qind_list
    # k_indices = [(kx, ky, kz) for kx in kind_list for ky in kind_list for kz in kind_list]
    # k_indices=gen_qindices(kind_list)
    # Parallelize the inner loop
    results = Parallel(n_jobs=cores_used)(delayed(precalcsig_innerloop)(k, qxind, qyind, qzind, knum, n, Pk, Gk,opt) for k in essential_kpoints)
    # Combine the results
    for i, k in enumerate(essential_kpoints):
        # kx, ky, kz = k
        kx=k[0]
        ky=k[1]
        kz=k[2]
        res=np.squeeze(results[i])

        all_sym_kpoints=sym_mapping(kx,ky,kz,knum)
        for kpoint in all_sym_kpoints:
            sig[:,kpoint[0],kpoint[1],kpoint[2]]=res*(kpoint[3]**power)
    return sig*-1*U*U / beta * ( 1/ a / knum) ** 3#2*np.pi


#----------test functions---------
# all tests are made in the trivial dispersion e_k=0.


def FT_test(quant,knum,a=1):
    k1,k2,k3=gen_full_kgrids(knum)
    kx=(-k1+k2+k3)*np.pi/a
    ky=(k1-k2+k3)*np.pi/a
    kz=(k1+k2-k3)*np.pi/a
    factor0=1
    factor1=np.exp(1j*kx+1j*ky)# remember to use complex unit cell!
    factor2=np.exp(1j*kx+2j*ky+1j*kz)
    # factor3=np.exp(3j*kx+3j*ky)
    quantR0=np.sum(quant*factor0,axis=(1,2,3))/knum**3
    quantR1=np.sum(quant*factor1,axis=(1,2,3))/knum**3
    quantR2=np.sum(quant*factor2,axis=(1,2,3))/knum**3
    # quantR3=np.sum(quant*factor3,axis=(1,2,3))/knum**3
    plt.plot(quantR0.real,label='local unit cell.real')
    plt.plot(quantR0.imag,label='local unit cell.imag')
    plt.plot(quantR1.real,label='1st NN unit cell.real')
    plt.plot(quantR1.imag,label='1st NN unit cell.imag')
    plt.plot(quantR2.real,label='2nd NN unit cell.real')
    plt.plot(quantR2.imag,label='2nd NN unit cell.imag')
    # plt.plot(quantR3,label='quantR(3,0,0)')
    plt.legend()
    plt.grid()
    plt.show()
    return 0


def G_test(sigA,sigB,U,T,knum,a=1):
    start_time = time.time()
    # U=4.0
    mu=U/2
    # T=0.2
    beta=1/T
    # sigma=np.loadtxt('./trial_sigma/{}_{}.dat'.format(U,T))[:500,:]
    # sigA=sigma[:,1]+1j*sigma[:,2]
    # sigB=sigma[:,3]+1j*sigma[:,4]
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

    kxind1=1
    kyind1=2
    kzind1=6
    kxind2=9
    kyind2=8
    kzind2=4
    k1,k2,k3=gen_full_kgrids(knum,a)
    dis=dispersion(k1, k2, k3)
    print(dis[kxind1,kyind1,kzind1],dis[kxind2,kyind2,kzind2])
    plt.plot(fermion_om[2*n:3*n],G12[2*n:3*n,kxind1,kyind1,kzind1].real,label='G12_k1_real')
    plt.plot(fermion_om[2*n:3*n],G12[2*n:3*n,kxind1,kyind1,kzind1].imag,label='G12_k1_imag')
    plt.plot(fermion_om[2*n:3*n],G12[2*n:3*n,kxind2,kyind2,kzind2].real,label='G12_k2_real')
    plt.plot(fermion_om[2*n:3*n],G12[2*n:3*n,kxind2,kyind2,kzind2].imag,label='G12_k2_imag')
    # plt.plot(fermion_om,G_A[:,knum-1-kxind,kyind,kzind].real,label='G-k_A_real')
    # plt.plot(fermion_om,G_A[:,knum-1-kxind,kyind,kzind].imag,label='G-k_A_imag')
    # plt.plot(fermion_om[2*n:3*n],G22[2*n:3*n,kxind,kyind,kzind].real,label='G22_k_real')
    # plt.plot(fermion_om[2*n:3*n],G22[2*n:3*n,kxind,kyind,kzind].imag,label='G22_k_imag')
    # plt.plot(fermion_om,G12[:,kxind,kyind,kzind].real,label='G12_k_real')
    # plt.plot(fermion_om,G12[:,kxind,kyind,kzind].imag,label='G12_k_imag')
    plt.legend()
    plt.grid()
    plt.show()

    # FT_test(G11[2*n:3*n],knum)
    # FT_test(G12[2*n:3*n],knum)
    return 0
#clear

def precalcP_test(sigA,sigB,U,T,knum,a=1):
    # U=6.0
    mu=U/2
    # T=0.3
    beta=1/T
    # sigma=np.loadtxt('{}_{}.dat'.format(U,T))[:500,:]
    # sigA=sigma[:,1]+1j*sigma[:,2]
    # sigB=sigma[:,3]+1j*sigma[:,4]
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
    qxind=1
    qyind=2
    qzind=3
    kxind=0
    kyind=0
    kzind=0

    start_time = time.time()
    #brute-force method: for P12=P21
    P12=precalcP12(beta,knum,G12)

    # new trick
    # P11=precalcP11(beta,knum,G11,allsigA,mu)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("time is {:.6f} s".format(elapsed_time))
    Boson_om = (2*np.arange(2*n+1)-2*n)*np.pi/beta
    # plt.plot(Boson_om,P11[:,qxind,qyind,qzind].real,label='P22_real')
    # plt.plot(Boson_om,P11[:,qxind,qyind,qzind].imag,label='P22_imag')
    # plt.plot(Boson_om,P12[:,qxind,qyind,qzind].real,label='P12_real')
    # plt.plot(Boson_om,P12[:,qxind,qyind,qzind].imag,label='P12_imag')
    # plt.plot(Boson_om,P12[:,kxind,kyind,kzind].real,label='P12_real')
    # plt.plot(Boson_om,P12[:,kxind,kyind,kzind].imag,label='P12_imag')
    # plt.legend()
    # plt.show()

    # FT_test(P11[n:2*n],knum)
    # k1,k2,k3=gen_full_kgrids(knum,a)
    # dis=dispersion(k1, k2, k3)
    # for kxind in np.arange(knum):
    #     for kyind in np.arange(knum):
    #         for kzind in np.arange(knum):

    #             # print('sign of dispk and sig_12k',dis[kxind,kyind,kzind]/np.abs(dis[kxind,kyind,kzind]),sig_new_12[500,kxind,kyind,kzind]/np.abs(sig_new_12[500,kxind,kyind,kzind]))
    #             if dis[kxind,kyind,kzind]*P12[500,kxind,kyind,kzind]>0:
    #                 print(kxind,kyind,kzind,dis[kxind,kyind,kzind],P12[500,kxind,kyind,kzind])
    return 0
#Clear.




def sig_imp_pert_test(sigA,sigB,U,T,knum):
    # U=2.0
    mu=U/2
    # T=0.01
    beta=1/T
    # sigma=np.loadtxt('{}_{}.dat'.format(U,T))[:500,:]
    # sigA=sigma[:,1]+1j*sigma[:,2]
    # sigB=sigma[:,3]+1j*sigma[:,4]
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
    G12=G_12(knum,z_A,z_B)
    # print(np.shape(G11),np.shape(G22))
    G11_imp=np.sum(G11,axis=(1,2,3))/knum**3
    G22_imp=np.sum(G22,axis=(1,2,3))/knum**3
    #indeed, here G12_imp=0.
    # G12_imp=np.sum(G22,axis=(1,2,3))/knum**3
    # plt.plot(G12_imp)
    # plt.show()
    sigimp11,sigimp22=perturb_imp.pertimp_func(G11_imp,G22_imp,delta_inf,beta,U,eps2_ave)
    return sigimp11,sigimp22

def new_sig(sigA,sigB,U,T,knum,a=1):
    print("doing perturbation......")
    start_time = time.time()
    mu=U/2
    beta=1/T
    n=sigA.size
    # sigma=np.loadtxt('{}_{}.dat'.format(U,T))[:n,:]
    # sigA=sigma[:,1]+1j*sigma[:,2]#sig+delta
    # sigB=sigma[:,3]+1j*sigma[:,4]#sig-delta
    z_A=z(beta,mu,sigA)#z-delta
    z_B=z(beta,mu,sigB)#z+delta
    # n=sigA.size
    allsigA=ext_sig(beta,sigA)
    allsigB=ext_sig(beta,sigB)
    sig_pert_imp11,sig_pert_imp22=sig_imp_pert_test(sigA,sigB,U,T,knum)
    # qxind=2
    # qyind=5
    # qzind=7
    # kxind=0
    # kyind=1
    # kzind=2

    G11=G_11(knum,z_A,z_B)
    G12=G_12(knum,z_A,z_B)
    time_G=time.time()
    print("Green's function finished. time to calculate prepare all G is {:.6f} s".format(time_G-start_time))
    P12=precalcP12(beta,knum,G12)
    P11=precalcP11(beta,knum,G11,allsigA,mu)
    Boson_om = (2*np.arange(2*n+1)-2*n)*np.pi/beta 
    # plt.plot(Boson_om,P11[:,qxind,qyind,qzind].real,label='P11q_real')
    # plt.plot(Boson_om,P11[:,qxind,qyind,qzind].imag,label='P11q_imag')
    # plt.plot(Boson_om,P11[:,kxind,kyind,kzind].real,label='P11k_real')
    # plt.plot(Boson_om,P11[:,kxind,kyind,kzind].imag,label='P11k_imag')
    # plt.grid()
    # plt.legend()
    # plt.show()
    # FT_test(P11[n:2*n],knum)  

    time_P=time.time()
    print("Polarization function finished.time to calculate prepare all G and P is {:.6f} s".format(time_P-time_G))


    sig_11=precalcsig(U,beta,knum,P11,G11,11)# actually P22 and G11. BUT P11=P22
    # FT_test(sig_11[n:2*n],knum)  
    sig_22=-sig_11.conjugate()
    sig_new_12=precalcsig(U,beta,knum,P12,G12,12)
    time_sig=time.time()
    print("time to calculate sigma is {:.6f} s".format(time_sig-time_P))
    # plt.plot(sig_pert_imp11[n:3*n].real,label='sigpert_imp11_real')
    # plt.plot(sig_pert_imp11[n:3*n].imag,label='sigpert_imp11_imag')
    # plt.plot(sig_11[n:3*n,kxind,kyind,kzind].real,label='sigk_pert_11_real')
    # plt.plot(sig_11[n:3*n,kxind,kyind,kzind].imag,label='sigk_pert_11_imag')
    # plt.plot(sig_11[n:3*n,qxind,qyind,qzind].real,label='sigq_pert_11_real')
    # plt.plot(sig_11[n:3*n,qxind,qyind,qzind].imag,label='sigq_pert_11_imag')
    # plt.grid()
    # plt.legend()
    # plt.show()
    # plt.plot(sig_22[n:3*n,kxind,kyind,kzind].real,label='sig_pert__22_real')
    # plt.plot(sig_22[n:3*n,kxind,kyind,kzind].imag,label='sig_pert__22_imag')
    # plt.plot(sig_new_12[:,kxind,kyind,kzind].real,label='sigk_12_real')
    # plt.plot(sig_new_12[:,kxind,kyind,kzind].imag,label='sigk_12_imag')
    # plt.plot(sig_new_12[:,qxind,qyind,qzind].real,label='sigq_12_real')
    # plt.plot(sig_new_12[:,qxind,qyind,qzind].imag,label='sigq_12_imag')
    # plt.grid()
    # plt.legend()
    # plt.show()
    # FT_test(G11[2*n:3*n],knum)  

    
    # plt.plot(sig_pert_imp11[n:3*n].real,label='sig_pert_imp_11_real')
    # plt.plot(sig_pert_imp11[n:3*n].imag,label='sig_pert_imp_11_imag')
    # plt.plot(sig_pert_imp22[n:3*n].real,label='sig_pert__imp_22_real')
    # plt.plot(sig_pert_imp22[n:3*n].imag,label='sig_pert__imp_22_imag')
    sig_new_11=allsigA[n:3*n, None, None, None]+sig_11-sig_pert_imp11[:, None, None, None]
    sig_new_22=allsigB[n:3*n, None, None, None]+sig_22-sig_pert_imp22[:, None, None, None]
    # plt.plot(sig_new_11[:,kxind,kyind,kzind].real,label='sig_new_11_real')
    # plt.plot(sig_new_11[:,kxind,kyind,kzind].imag,label='sig_new_11_imag')
    # plt.plot(sig_new_22[:,kxind,kyind,kzind].real,label='sig_new_22_real')
    # plt.plot(sig_new_22[:,kxind,kyind,kzind].imag,label='sig_new_22_imag')
    # plt.plot(sig_new_12[:,kxind,kyind,kzind].real,label='sig_new_12_real')
    # plt.plot(sig_new_12[:,kxind,kyind,kzind].imag,label='sig_new_12_imag')
    # plt.legend()
    # plt.show()
    # k1,k2,k3=gen_full_kgrids(knum,a)
    # dis=dispersion(k1, k2, k3)
    # for kxind in np.arange(knum):
    #     for kyind in np.arange(knum):
    #         for kzind in np.arange(knum):

    #             # print('sign of dispk and sig_12k',dis[kxind,kyind,kzind]/np.abs(dis[kxind,kyind,kzind]),sig_new_12[500,kxind,kyind,kzind]/np.abs(sig_new_12[500,kxind,kyind,kzind]))
    #             if dis[kxind,kyind,kzind]*sig_new_12[500,kxind,kyind,kzind]<0:
    #                 print(kxind,kyind,kzind,dis[kxind,kyind,kzind],sig_new_12[500,kxind,kyind,kzind])
    return sig_new_11,sig_new_22,sig_new_12
#clear. 

def impurity_test(SigA,SigB,U,T,knum):
    # U=8.0
    # T=0.42
    mu=U/2
    # knum=10
    beta=1/T
    n=SigA.size

    iom= 1j*(2*np.arange(2*n)+1-2*n)*np.pi/beta
    fermion_om=(2*np.arange(n)+1)*np.pi/beta
    # to generate dispertion 
    disp=calc_disp(knum)


    # just for test. old sigma.
    # allsigA=ext_sig(beta,SigA)[n:3*n]
    # allsigB=ext_sig(beta,SigB)[n:3*n]
    # Gk_11=(iom[:, None, None, None]+mu-allsigA[:, None, None, None])/((iom[:, None, None, None]+mu-allsigA[:, None, None, None])*(iom[:, None, None, None]+mu-allsigB[:, None, None, None])-(disp[None, :, :, :])**2)
    # Gk_22=(iom[:, None, None, None]+mu-allsigB[:, None, None, None])/((iom[:, None, None, None]+mu-allsigA[:, None, None, None])*(iom[:, None, None, None]+mu-allsigB[:, None, None, None])-(disp[None, :, :, :])**2)
    # Gk_imp_11=np.sum(Gk_11,axis=(1,2,3))/knum**3
    # Gk_imp_22=np.sum(Gk_22,axis=(1,2,3))/knum**3
    # plt.plot(fermion_om,Gk_imp_11[n:2*n].real,label='Gk_imp_11 real')
    # plt.plot(fermion_om,Gk_imp_11[n:2*n].imag,label='Gk_imp_11 imag')
    # plt.plot(fermion_om,Gk_imp_22[n:2*n].real,label='Gk_imp_22 real')
    # plt.plot(fermion_om,Gk_imp_22[n:2*n].imag,label='Gk_imp_22 imag')
    # plt.plot(Gk_imp_12[n:2*n].real,label='Gk_imp_12 real')
    # plt.plot(Gk_imp_12[n:2*n].imag,label='Gk_imp_12 imag')
    # plt.legend()
    # plt.grid()
    # plt.show()
    # Delta0_11=iom+mu-allsigB-1/Gk_imp_11
    # Delta0_22=iom+mu-allsigA-1/Gk_imp_22
    # plt.plot(fermion_om,Delta0_11[n:2*n].real,label='Delta0_11 real')
    # plt.plot(fermion_om,Delta0_11[n:2*n].imag,label='Delta0_11 imag')
    # plt.plot(fermion_om,Delta0_22[n:2*n].real,label='Delta0_22 real')
    # plt.plot(fermion_om,Delta0_22[n:2*n].imag,label='Delta0_22 imag')
    # plt.legend()
    # plt.grid()
    # plt.show()
    # return 0
    # end of test


    sig_new_11,sig_new_22,sig_new_12=new_sig(SigA,SigB,U,T,knum,n)
    print("perturbed sigma finished!")
    sig_imp_new_11=np.sum(sig_new_11,axis=(1,2,3))/knum**3
    sig_imp_new_22=np.sum(sig_new_22,axis=(1,2,3))/knum**3
    print("impurity perturbed sigma finished!")

    # output test
    # fnewsig='test_new_sig.imp'
    # f = open(fnewsig, 'w')
    # for i,iom in enumerate(fermion_om):
    #     print(iom, sig_imp_new_11[i].real, sig_imp_new_11[i].imag, sig_imp_new_22[i].real, sig_imp_new_22[i].imag, file=f) 
    # f.close()


    # plt.plot(fermion_om,sig_imp_new_11[n:2*n].real,label='sig_imp_new_11 real')
    # plt.plot(fermion_om,sig_imp_new_11[n:2*n].imag,label='sig_imp_new_11 imag')
    # plt.plot(fermion_om,sig_imp_new_22[n:2*n].real,label='sig_imp_new_22 real')
    # plt.plot(fermion_om,sig_imp_new_22[n:2*n].imag,label='sig_imp_new_22 imag')

    # plt.legend()
    # plt.grid()
    # plt.show()
    Gk_new_11=(iom[:, None, None, None]+mu-sig_new_11)/((iom[:, None, None, None]+mu-sig_new_11)*(iom[:, None, None, None]+mu-sig_new_22)-(disp[None, :, :, :]+sig_new_12)**2)#
    Gk_new_22=(iom[:, None, None, None]+mu-sig_new_22)/((iom[:, None, None, None]+mu-sig_new_11)*(iom[:, None, None, None]+mu-sig_new_22)-(disp[None, :, :, :]+sig_new_12)**2)#
    # Gk_new_12=(disp[None, :, :, :]+sig_new_12)/((iom[:, None, None, None]+mu-sig_new_11)*(iom[:, None, None, None]+mu-sig_new_22)-(disp[None, :, :, :]+sig_new_12)**2)#
    # max_sym_index,essential_kpoints, sym_array=calc_sym_array(knum)
    # for points in essential_kpoints:

    #     plt.plot(fermion_om,sig_new_12[n:2*n,points[0],points[1],points[2]].real,label='sig_new_12[{},{},{}] real'.format(points[0],points[1],points[2]))
    #     plt.plot(fermion_om,sig_new_12[n:2*n,points[0],points[1],points[2]].imag,label='sig_new_12[{},{},{}] imag'.format(points[0],points[1],points[2]))
    #     plt.legend()
    #     plt.grid()
    #     plt.show()
    
    Gk_imp_new_11=np.sum(Gk_new_11,axis=(1,2,3))/knum**3
    Gk_imp_new_22=np.sum(Gk_new_22,axis=(1,2,3))/knum**3
    # Gk_imp_new_12=np.sum(Gk_new_12,axis=(1,2,3))/knum**3
    print("perturbed Green's functions finished!")
    # plt.plot(fermion_om,Gk_imp_new_11[n:2*n].real,label='Gk_imp_new_11 real')
    # plt.plot(fermion_om,Gk_imp_new_11[n:2*n].imag,label='Gk_imp_new_11 imag')
    # plt.plot(fermion_om,Gk_imp_new_22[n:2*n].real,label='Gk_imp_new_22 real')
    # plt.plot(fermion_om,Gk_imp_new_22[n:2*n].imag,label='Gk_imp_new_22 imag')
    # plt.legend()
    # plt.grid()
    # plt.show()
    Delta_11=iom+mu-sig_imp_new_11-1/Gk_imp_new_22
    Delta_22=iom+mu-sig_imp_new_22-1/Gk_imp_new_11
    print("perturbed hybridization functions finished!")
    # plt.plot(fermion_om,Delta_11[n:2*n].real,label='Delta_11 real')
    # plt.plot(fermion_om,Delta_11[n:2*n].imag,label='Delta_11 imag')
    # plt.plot(fermion_om,Delta_22[n:2*n].real,label='Delta_22 real')
    # plt.plot(fermion_om,Delta_22[n:2*n].imag,label='Delta_22 imag')
    # plt.legend()
    # plt.grid()
    # plt.show()
    # Deltatau_11=np.fft.ifft(Delta_11)
    # Deltatau_22=np.fft.ifft(Delta_22)
    # plt.plot(Deltatau_11.real,label='Deltatau_11 real')
    # plt.plot(Deltatau_11.imag,label='Deltatau_11 imag')
    # plt.plot(Deltatau_22.real,label='Deltatau_22 real')
    # plt.plot(Deltatau_22.imag,label='Deltatau_22 imag')
    # plt.legend()
    # plt.grid()
    # plt.show()
    return Delta_11[n:2*n],Delta_22[n:2*n]


#when call functions in this file, comment everything below!

# T=0.2
# U=4.0
# knum=10
# nfreq=500

# if (len(sys.argv)!=3):
#     print('usually we need 2 parameters:T and U.')
    
# if (len(sys.argv)==3):
#     U=float(sys.argv[1])
#     T=float(sys.argv[2])
#     print('T=',T)
#     print('U=',U)

# sigma=np.loadtxt('./trial_sigma/{}_{}.dat'.format(U,T))[:nfreq,:]
# sigA=sigma[:,1]+1j*sigma[:,2]#sig+delta
# sigB=sigma[:,3]+1j*sigma[:,4]#sig-delta

# sym_mapping(1,2,3)
# calc_sym_array(10)
# G_test(sigA,sigB,U,T,knum)
# precalcP_test(sigA,sigB,U,T,knum)
# sig_imp_pert_test(sigA,sigB,U,T,2*knum)
# new_sig(U,T,knum,500)
# impurity_test(sigA,sigB,U,T,knum)
