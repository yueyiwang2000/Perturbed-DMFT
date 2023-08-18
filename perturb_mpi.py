import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess
import time
from joblib import Parallel, delayed
import perturb_imp
import hilbert
from mpi4py import MPI
# cores_used=8
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

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

def calc_expikdel(knum):
    k1,k2,k3=gen_full_kgrids(knum)
    phase=1j*(k1+k2-k3)*np.pi# Delta=0.5*(a1+a2-a3), ai*ki=2Pi, ai*ki=0
    factor=np.exp(phase)
    return factor

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
def precalcP12_innerloop(q, knum, n, G12):
    # start_time = time.time()
    qx=q[0]
    qy=q[1]
    qz=q[2]
    kind_list = np.arange(knum)
    kxind, kyind, kzind = np.meshgrid(kind_list, kind_list, kind_list, indexing='ij')
    # q_time = time.time()
    # P_partial = np.zeros(2*n+1)
    P_partial = np.zeros(n+1)
    G_12_factor=(-1)**((np.mod(kxind + qx, knum)-(kxind+qx))/knum+(np.mod(kyind + qy, knum)-(kyind+qy))/knum+(np.mod(kzind + qz, knum)-(kzind+qz))/knum)
    # factor_time = time.time()
    G12_kq = G_12_factor*G12[:, np.mod(kxind + qx, knum), np.mod(kyind + qy, knum), np.mod(kzind + qz, knum)]
    G12_sliced = G12[:3*n]
    slice_time=time.time()
    for Omind in np.arange(n+1)+n:#:
        # start_time = time.time()
        G12_kq_sliced=G12_kq[Omind-n:3*n+Omind-n]
        # slicetime=time.time()
        Gmul=G12_sliced * G12_kq_sliced# takes most time
        # multime=time.time()
        P_partial[2*n-Omind] = np.sum(Gmul)
        # sumtime=time.time()
        # P_partial[2*n-Omind]=P_partial[Omind]   
    return P_partial

def precalcP12_mpi(beta, knum, G1, a=1):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    n = int(np.shape(G1)[0] / 4)
    max_sym_index,essential_kpoints, sym_array=calc_sym_array(knum)
    # print('max_sym_index=',max_sym_index)
    if knum % 2 != 0:
            print('knum should be a even number!')
            return 0
    if rank ==0:
        # devide max_sym_index for nproc processors
        ave, res = divmod(max_sym_index, nprocs)
        counts = [ave + 1 if p < res else ave for p in range(nprocs)]
        starts = [sum(counts[:p]) for p in range(nprocs)]
        ends = [sum(counts[:p+1]) for p in range(nprocs)]
        qpoints = [(starts[p], ends[p]) for p in range(nprocs)]
        # print(qpoints)
    else:
        qpoints=None
    qpoints=comm.scatter(qpoints, root=0)
    # print(qpoints)
    # assign q points for different procs.
    pointsperproc=int(max_sym_index/nprocs)+1
    partP = np.zeros((pointsperproc,n+1),dtype=np.float64)
    for qind in np.arange(qpoints[1]-qpoints[0]):
        q=essential_kpoints[qind+pointsperproc*rank]
        # print('calculating P12.Process {} has qpoint:'.format(rank), q)
        partP[qind,:]=precalcP12_innerloop(q, knum, n, G1)


    gathered_P=np.zeros((nprocs,pointsperproc,n+1),dtype=np.float64)
    comm.Gather(partP,gathered_P,root=0)
    # gathered_P = comm.gather(partP, root=0)
    if rank==0:
        # P is compacted for fast connection between procs. now unpack it:
        full_P=np.zeros((2*n+1, knum, knum, knum),dtype=float)
        for proc in np.arange(nprocs):
            for ind in np.arange(pointsperproc):
                qind=proc*pointsperproc+ind
                if qind < max_sym_index:
                    q=essential_kpoints[qind]
                    full_P[:n+1,q[0],q[1],q[2]]=gathered_P[proc,ind,:]
                    # restore k-space domain sym
                    all_sym_kpoints=sym_mapping(q[0],q[1],q[2],knum)
                    for kpoint in all_sym_kpoints:
                        full_P[:,kpoint[0],kpoint[1],kpoint[2]]=full_P[:,q[0],q[1],q[2]]*kpoint[3]
        # restore freq domain sym
        full_P[n+1:,:,:,:]=full_P[n-1::-1,:,:,:]
        return full_P / beta * (1/ a / knum) ** 3#2 * np.pi 
    # else:
    #     MPI.Finalize()
    #     exit()

def fermi(eps,beta):
    return 1/(np.exp(beta*eps)+1)

# This is for diagonal elements for polarization. Here the only way to solve the convergence issue
# is that to contour intregration in brute force, which will result in 4 terms. 
# here alpha_k=sqrt(delta_inf**2+eps_k**2)
def precalcP11_innerloop(q, knum, n, G_k,G_k0,alphak,f_alphak,deltainf,beta):
    qx=q[0]
    qy=q[1]
    qz=q[2]
    kind_list = np.arange(knum)
    kxind, kyind, kzind = np.meshgrid(kind_list, kind_list, kind_list, indexing='ij')
    P_partial = np.zeros((n+1), dtype=complex)
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
        P_partial[2*n-Omind] =  +beta*np.sum(lindhard)+np.sum(Gk_slice * G_kq[Omind-n:3*n+Omind-n]-Gk0_slice * G_kq0[Omind-n:3*n+Omind-n])
        P_partial[2*n-Omind]=P_partial[2*n-Omind].conjugate()
        # 
        #brute-force 
        # P_partial[Omind, 0, 0, 0]=np.sum(G_k[n:3*n] * G_kq[n+Omind-n:3*n+Omind-n])
        # P_partial[2*n-Omind, 0, 0, 0]=P_partial[Omind, 0, 0, 0].conjugate()
    return P_partial.real

def precalcP11_mpi(beta, knum, Gk, fullsig,mu,a=1):# here, we need original sigma.
    n = int(np.shape(Gk)[0] / 4)
    if knum % 2 != 0:
        print('knum should be a even number!')
        return 0

    #preparation for this trick. should we only do these in 1 proc and bcast to other procs?
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

    # parallelization
    max_sym_index,essential_kpoints, sym_array=calc_sym_array(knum)
    if rank ==0:
        # devide max_sym_index for nproc processors
        ave, res = divmod(max_sym_index, nprocs)
        counts = [ave + 1 if p < res else ave for p in range(nprocs)]
        starts = [sum(counts[:p]) for p in range(nprocs)]
        ends = [sum(counts[:p+1]) for p in range(nprocs)]
        qpoints = [(starts[p], ends[p]) for p in range(nprocs)]
        # print(qpoints)
    else:
        qpoints=None
    qpoints=comm.scatter(qpoints, root=0)
    # print(qpoints)

    # assign q points for different procs. and calculate
    pointsperproc=int(max_sym_index/nprocs)+1
    partP = np.zeros((pointsperproc,n+1),dtype=np.complex128)
    for qind in np.arange(qpoints[1]-qpoints[0]):
        q=essential_kpoints[qind+pointsperproc*rank]
        # print('calculating P11.Process {} has qpoint:'.format(rank), q)
        partP[qind,:]=precalcP11_innerloop(q, knum, n, Gk,Gk0,alphak,f_alphak,delta_inf,beta)
    
    gathered_P=np.zeros((nprocs,pointsperproc,n+1),dtype=np.complex128)
    comm.Gather(partP,gathered_P,root=0)
    # gathered_P = comm.gather(partP, root=0)
    if rank==0:
        # P is compacted for fast connection between procs. now unpack it:
        full_P=np.zeros((2*n+1, knum, knum, knum),dtype=np.complex128)
        for proc in np.arange(nprocs):
            for ind in np.arange(pointsperproc):
                qind=proc*pointsperproc+ind
                if qind < max_sym_index:
                    q=essential_kpoints[qind]
                    full_P[:n+1,q[0],q[1],q[2]]=gathered_P[proc,ind,:]
        # restore k-space domain sym
                    all_sym_kpoints=sym_mapping(q[0],q[1],q[2],knum)
                    for kpoint in all_sym_kpoints:
                        full_P[:,kpoint[0],kpoint[1],kpoint[2]]=full_P[:,q[0],q[1],q[2]]
        # restore freq domain sym
        full_P[n+1:,:,:,:]=full_P[n-1::-1,:,:,:].conjugate()
        return full_P / beta * (1/ a / knum) ** 3

# Note: remember G and P has different spin.
#since every sigma_k can be only used once, seems we don't need to store them in RAM.
# However, for a specific k point, we still want sigma at every fermion matsubara freq.

def precalcsig_innerloop(k,knum, n, P_k, Gk,opt):
    kx=k[0]
    ky=k[1]
    kz=k[2]
    qind_list = np.arange(knum)
    qxind, qyind, qzind = np.meshgrid(qind_list, qind_list, qind_list, indexing='ij')

    if opt==12:
        G_12_factor=(-1)**((np.mod(qxind + kx, knum)-(qxind+kx))/knum+(np.mod(qyind + ky, knum)-(qyind+ky))/knum+(np.mod(qzind + kz, knum)-(qzind+kz))/knum)
    elif opt==11:
        G_12_factor=1
    else:
        print('please specify 12 or 11!')
        return 0
    sig_partial = np.zeros((n), dtype=np.complex128)
    G_kq = G_12_factor*Gk[:, np.mod(qxind + kx, knum), np.mod(qyind + ky, knum), np.mod(qzind + kz, knum)]
    for omind in np.arange(n):
        sig_partial[omind] = np.sum(P_k * G_kq[omind:omind +2*n+1])# from omind, omind+1, ..., to omind+2n
        # sig_partial[2*n-1-omind, 0, 0, 0]=sig_partial[omind, 0, 0, 0].conjugate()
    # print(k,sig_partial[n,0,0,0])
    # plt.plot(sig_partial[n:3*n,0,0,0].real,label='sigk_pert_11_real')
    # plt.plot(sig_partial[n:3*n,0,0,0].imag,label='sigk_pert_11_imag')
    # plt.grid()
    # plt.legend()
    # plt.show()
    return sig_partial

def precalcsig_mpi(U,beta, knum, Pk, Gk, opt,a=1):
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

    max_sym_index,essential_kpoints, sym_array=calc_sym_array(knum)
    if rank ==0:
        # devide max_sym_index for nproc processors
        ave, res = divmod(max_sym_index, nprocs)
        counts = [ave + 1 if p < res else ave for p in range(nprocs)]
        starts = [sum(counts[:p]) for p in range(nprocs)]
        ends = [sum(counts[:p+1]) for p in range(nprocs)]
        qpoints = [(starts[p], ends[p]) for p in range(nprocs)]
        # print(qpoints)
    else:
        qpoints=None
    qpoints=comm.scatter(qpoints, root=0)
    # print(qpoints)

    # assign q points for different procs. and calculate
    pointsperproc=int(max_sym_index/nprocs)+1
    partsig = np.zeros((pointsperproc,n),dtype=np.complex128)
    for qind in np.arange(qpoints[1]-qpoints[0]):
        q=essential_kpoints[qind+pointsperproc*rank]
        # print('calculating sig.Process {} has kpoint:'.format(rank), q)
        partsig[qind,:]=precalcsig_innerloop(q, knum, n, Pk,Gk,opt)
    gathered_sig=np.zeros((nprocs,pointsperproc,n),dtype=np.complex128)
    comm.Gather(partsig,gathered_sig,root=0)
    if rank==0:
        # P is compacted for fast connection between procs. now unpack it:
        full_sig=np.zeros((2*n, knum, knum, knum),dtype=np.complex128)
        for proc in np.arange(nprocs):
            for ind in np.arange(pointsperproc):
                qind=proc*pointsperproc+ind
                if qind < max_sym_index:
                    q=essential_kpoints[qind]
                    full_sig[:n,q[0],q[1],q[2]]=gathered_sig[proc,ind,:]
        # restore k-space domain sym
                    all_sym_kpoints=sym_mapping(q[0],q[1],q[2],knum)
                    for kpoint in all_sym_kpoints:
                        full_sig[:,kpoint[0],kpoint[1],kpoint[2]]=full_sig[:,q[0],q[1],q[2]]*(kpoint[3]**power)
        # restore freq domain sym
        full_sig[n:,:,:,:]=full_sig[n-1::-1,:,:,:].conjugate()
        return full_sig*-1*U*U / beta * ( 1/ a / knum) ** 3#2*np.pi


#----------test functions---------


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

def Delta_DMFT(sigA,sigB,U,T,knum,a=1):
    mu=U/2
    beta=1/T
    n=sigA.size
    om= (2*np.arange(n)+1)*np.pi/beta
    iom=1j*om
    z_A=z(beta,mu,sigA)
    z_B=z(beta,mu,sigB)
    G11=G_11(knum,z_A,z_B)[2*n:3*n]
    G22=-G11.conjugate()
    G12=G_12(knum,z_A,z_B)[2*n:3*n]
    factor=calc_expikdel(knum)
    actual_G12=G12*factor
    G12_imp=np.sum(actual_G12,axis=(1,2,3))/knum**3# and, G12=G21.
    G11_imp=np.sum(G11,axis=(1,2,3))/knum**3
    G22_imp=np.sum(G22,axis=(1,2,3))/knum**3
    Gimp_inv_11=G22_imp/(G11_imp*G22_imp-G12_imp**2)
    Gimp_inv_22=G11_imp/(G11_imp*G22_imp-G12_imp**2)
    # print(np.shape(iom),np.shape(sigA),np.shape(Gimp_inv_11))
    Delta_11=iom+mu-sigA-Gimp_inv_11
    Delta_22=iom+mu-sigB-Gimp_inv_22
    return Delta_11,Delta_22

def G_test(sigA,sigB,U,T,knum,a=1):
    start_time = time.time()
    mu=U/2
    beta=1/T
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
    kzind1=3
    k1,k2,k3=gen_full_kgrids(knum,a)
    dis=dispersion(k1, k2, k3)
    factor=calc_expikdel(knum)
    actual_G12=G12*factor
    G12_imp=np.sum(actual_G12,axis=(1,2,3))/knum**3
    anaG12=0.5*(1/(1j*fermion_om[:,None,None,None]-dis[None,kxind1,kyind1,kzind1])-1/(1j*fermion_om[:,None,None,None]+dis[None,kxind1,kyind1,kzind1]))
    # print(dis[kxind1,kyind1,kzind1],dis[kxind2,kyind2,kzind2])
    # plt.plot(fermion_om[2*n:3*n],G12_imp[2*n:3*n].real,label='G12_imp real')
    # plt.plot(fermion_om[2*n:3*n],G12_imp[2*n:3*n].imag,label='G12_imp imag')
    # plt.plot(fermion_om[2*n:3*n],G12[2*n:3*n,kxind1,kyind1,kzind1].real,label='G12 real')
    # plt.plot(fermion_om[2*n:3*n],G12[2*n:3*n,kxind1,kyind1,kzind1].imag,label='G12 imag')
    # plt.plot(fermion_om[2*n:3*n],anaG12[2*n:3*n,0,0,0].real,label='G12_ana real')
    # plt.plot(fermion_om[2*n:3*n],anaG12[2*n:3*n,0,0,0].imag,label='G12_ana imag')
    # plt.legend()
    # plt.grid()
    # plt.show()
    # FT_test(G11[2*n:3*n],knum)
    return 0
#clear

def precalcP_test(sigA,sigB,U,T,knum,a=1):
    mu=U/2
    beta=1/T
    z_A=z(beta,mu,sigA)
    z_B=z(beta,mu,sigB)
    n=sigA.size
    allsigA=ext_sig(beta,sigA)
    allsigB=ext_sig(beta,sigB)
    G11=G_11(knum,z_A,z_B)
    # G22=-G11.conjugate()
    G12=G_12(knum,z_A,z_B)

    start_time = time.time()
    # P12=precalcP12_mpi(beta,knum,G12)
    P11=precalcP11_mpi(beta,knum,G11,allsigA,mu)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("time is {:.6f} s".format(elapsed_time))
    Boson_om = (2*np.arange(2*n+1)-2*n)*np.pi/beta

    # FT_test(G11[2*n:3*n],knum)
    # FT_test(P11[n:2*n],knum)
    # k1,k2,k3=gen_full_kgrids(knum,a)
    # dis=dispersion(k1, k2, k3)
    # for kxind in np.arange(knum):
    #     for kyind in np.arange(knum):
    #         for kzind in np.arange(knum):
    #             plt.plot(Boson_om,P11[:,kxind,kyind,kzind].real,label='P22_real')
    #             plt.plot(Boson_om,P11[:,kxind,kyind,kzind].imag,label='P22_imag')
                # plt.plot(Boson_om,P12[:,kxind,kyind,kzind].real,label='P12_real')
                # plt.plot(Boson_om,P12[:,kxind,kyind,kzind].imag,label='P12_imag')
                # plt.legend()
                # plt.show()
    return 0
#Clear.

def sig_imp_pert_test(sigA,sigB,U,T,knum):
    mu=U/2
    beta=1/T
    z_A=z(beta,mu,sigA)
    z_B=z(beta,mu,sigB)
    n=sigA.size
    allsigA=ext_sig(beta,sigA)
    allsigB=ext_sig(beta,sigB)
    delta_inf=np.abs(-mu+allsigA[-1].real)
    eps2_ave=calc_eps2_ave(knum)
    G11=G_11(knum,z_A,z_B)
    G22=-G11.conjugate()
    G12=G_12(knum,z_A,z_B)
    G11_imp=np.sum(G11,axis=(1,2,3))/knum**3
    G22_imp=np.sum(G22,axis=(1,2,3))/knum**3
    sigimp11,sigimp22=perturb_imp.pertimp_func(G11_imp,G22_imp,delta_inf,beta,U,eps2_ave)
    return sigimp11,sigimp22

def new_sig(sigA,sigB,U,T,knum,a=1):
    # print("doing perturbation......")
    # start_time = time.time()
    mu=U/2
    beta=1/T
    n=sigA.size
    z_A=z(beta,mu,sigA)#z-delta
    z_B=z(beta,mu,sigB)#z+delta
    allsigA=ext_sig(beta,sigA)
    allsigB=ext_sig(beta,sigB)
    G11=G_11(knum,z_A,z_B)
    G12=G_12(knum,z_A,z_B)
    P12=precalcP12_mpi(beta,knum,G12)
    P11=precalcP11_mpi(beta,knum,G11,allsigA,mu)
    if rank !=0:
        P12=np.zeros((2*n+1,knum,knum,knum),dtype=float)
        P11=np.zeros((2*n+1,knum,knum,knum),dtype=complex)
    comm.Bcast(P12, root=0)
    comm.Bcast(P11, root=0)
    Boson_om = (2*np.arange(2*n+1)-2*n)*np.pi/beta 
    sig_11=precalcsig_mpi(U,beta,knum,P11,G11,11)# actually P22 and G11. BUT P11=P22
    sig_new_12=precalcsig_mpi(U,beta,knum,P12,G12,12)
    if rank ==0:
        sig_22=-sig_11.conjugate()
        sig_pert_imp11,sig_pert_imp22=sig_imp_pert_test(sigA,sigB,U,T,knum)
        sig_new_11=allsigA[n:3*n, None, None, None]+sig_11-sig_pert_imp11[:, None, None, None]
        sig_new_22=allsigB[n:3*n, None, None, None]+sig_22-sig_pert_imp22[:, None, None, None]
    # FT_test(G11[2*n:3*n],knum)
    # FT_test(P11[n:2*n],knum)
    # FT_test(sig_11[n:2*n],knum)
    # for kxind in np.arange(knum):
    #     for kyind in np.arange(knum):
    #         for kzind in np.arange(knum):
    #             plt.plot(sig_11[:, kxind, kyind, kzind].real-sig_pert_imp11.real,label='real')
    #             plt.plot(sig_11[:, kxind, kyind, kzind].imag-sig_pert_imp11.imag,label='imag')
    #             plt.legend()
    #             plt.grid()
    #             plt.show()
        return sig_new_11,sig_new_22,sig_new_12
    else:
        MPI.Finalize()
        exit()

#clear. 

def Delta_pert_DMFT(SigA,SigB,U,T,knum):
    mu=U/2
    beta=1/T
    n=SigA.size

    iom= 1j*(2*np.arange(2*n)+1-2*n)*np.pi/beta
    fermion_om=(2*np.arange(n)+1)*np.pi/beta
    # to generate dispertion 
    disp=calc_disp(knum)


    # just for test. without perturbation.
    allsigA=ext_sig(beta,SigA)[n:3*n]
    allsigB=ext_sig(beta,SigB)[n:3*n]
    Gk_11=(iom[:, None, None, None]+mu-allsigA[:, None, None, None])/((iom[:, None, None, None]+mu-allsigA[:, None, None, None])*(iom[:, None, None, None]+mu-allsigB[:, None, None, None])-(disp[None, :, :, :])**2)
    Gk_22=(iom[:, None, None, None]+mu-allsigB[:, None, None, None])/((iom[:, None, None, None]+mu-allsigA[:, None, None, None])*(iom[:, None, None, None]+mu-allsigB[:, None, None, None])-(disp[None, :, :, :])**2)
    Gk_imp_11=np.sum(Gk_11,axis=(1,2,3))/knum**3
    Gk_imp_22=np.sum(Gk_22,axis=(1,2,3))/knum**3
    # plt.plot(fermion_om,Gk_imp_11[n:2*n].real,label='Gk_imp_11 real')
    # plt.plot(fermion_om,Gk_imp_11[n:2*n].imag,label='Gk_imp_11 imag')
    # plt.plot(fermion_om,Gk_imp_22[n:2*n].real,label='Gk_imp_22 real')
    # plt.plot(fermion_om,Gk_imp_22[n:2*n].imag,label='Gk_imp_22 imag')
    # plt.plot(Gk_imp_12[n:2*n].real,label='Gk_imp_12 real')
    # plt.plot(Gk_imp_12[n:2*n].imag,label='Gk_imp_12 imag')
    # plt.legend()
    # plt.grid()
    # plt.show()
    Delta0_11=iom+mu-allsigB-1/Gk_imp_11
    Delta0_22=iom+mu-allsigA-1/Gk_imp_22
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
    # print("perturbed sigma finished!")

    sig_imp_new_11=np.sum(sig_new_11,axis=(1,2,3))/knum**3
    sig_imp_new_22=np.sum(sig_new_22,axis=(1,2,3))/knum**3
    # print("impurity perturbed sigma finished!")

    # plt.plot(fermion_om,sig_imp_new_11[n:2*n].real-allsigA[n:2*n].real,label='sig_imp_new_11 real')
    # plt.plot(fermion_om,sig_imp_new_11[n:2*n].imag-allsigA[n:2*n].imag,label='sig_imp_new_11 imag')
    # plt.plot(fermion_om,sig_imp_new_22[n:2*n].real-allsigB[n:2*n].real,label='sig_imp_new_22 real')
    # plt.plot(fermion_om,sig_imp_new_22[n:2*n].imag-allsigB[n:2*n].imag,label='sig_imp_new_22 imag')
    # plt.legend()
    # plt.grid()
    # plt.show()
    Gk_new_11=(iom[:, None, None, None]+mu-sig_new_22)/((iom[:, None, None, None]+mu-sig_new_11)*(iom[:, None, None, None]+mu-sig_new_22)-(-1*disp[None, :, :, :]+sig_new_12)**2)#
    Gk_new_22=(iom[:, None, None, None]+mu-sig_new_11)/((iom[:, None, None, None]+mu-sig_new_11)*(iom[:, None, None, None]+mu-sig_new_22)-(-1*disp[None, :, :, :]+sig_new_12)**2)#
    Gk_new_12=(-1*disp[None, :, :, :]+sig_new_12)/((iom[:, None, None, None]+mu-sig_new_11)*(iom[:, None, None, None]+mu-sig_new_22)-(disp[None, :, :, :]+sig_new_12)**2)#
    # max_sym_index,essential_kpoints, sym_array=calc_sym_array(knum)
    # for points in essential_kpoints:

    #     plt.plot(fermion_om,sig_new_12[n:2*n,points[0],points[1],points[2]].real,label='sig_new_12[{},{},{}] real'.format(points[0],points[1],points[2]))
    #     plt.plot(fermion_om,sig_new_12[n:2*n,points[0],points[1],points[2]].imag,label='sig_new_12[{},{},{}] imag'.format(points[0],points[1],points[2]))
    #     plt.legend()
    #     plt.grid()
    #     plt.show()
    
    Gk_imp_new_11=np.sum(Gk_new_11,axis=(1,2,3))/knum**3
    Gk_imp_new_22=np.sum(Gk_new_22,axis=(1,2,3))/knum**3
    factor=calc_expikdel(knum)
    Gk_imp_new_12=np.sum(Gk_new_12*factor,axis=(1,2,3))/knum**3
    # print("perturbed Green's functions finished!")
    # plt.plot(fermion_om,Gk_imp_new_11[n:2*n].real,label='Gk_imp_new_11 real')
    # plt.plot(fermion_om,Gk_imp_new_11[n:2*n].imag,label='Gk_imp_new_11 imag')
    # plt.plot(fermion_om,Gk_imp_new_22[n:2*n].real,label='Gk_imp_new_22 real')
    # plt.plot(fermion_om,Gk_imp_new_22[n:2*n].imag,label='Gk_imp_new_22 imag')
    # plt.legend()
    # plt.grid()
    # plt.show()
    # which part gives the most contribution to the delta?

    #full perturbation delta
    Delta_11=iom+mu-sig_imp_new_11-1/Gk_imp_new_11
    Delta_22=iom+mu-sig_imp_new_22-1/Gk_imp_new_22
    # plt.plot(Delta_11[n:2*n].imag,label='perturbed DMFT')

    # # without sig12 off diagonal term
    # Gk1_new_11=(iom[:, None, None, None]+mu-sig_new_22)/((iom[:, None, None, None]+mu-sig_new_11)*(iom[:, None, None, None]+mu-sig_new_22)-(disp[None, :, :, :])**2)#
    # Gk1_imp_new_11=np.sum(Gk1_new_11,axis=(1,2,3))/knum**3
    # Delta1_11=iom+mu-sig_imp_new_11-1/Gk1_imp_new_11
    # plt.plot(fermion_om,Delta1_11[n:2*n].imag,label='Delta1_11 imag')

    # use original sigma_imp in z
    # Delta2_11=iom[n:2*n]+mu-SigA-1/Gk_imp_new_11[n:2*n]
    # plt.plot(fermion_om,Delta2_11.imag,label='Delta2_11 imag')

    #use original sigma_DMFT_11 in Gk11
    # Gk3_new_11=(iom[:, None, None, None]+mu-allsigB[:, None, None, None])/((iom[:, None, None, None]+mu-allsigA[:, None, None, None])*(iom[:, None, None, None]+mu-allsigB[:, None, None, None])-(-1*disp[None, :, :, :]+sig_new_12)**2)#
    # Gk3_imp_new_11=np.sum(Gk3_new_11,axis=(1,2,3))/knum**3
    # Delta3_11=iom+mu-sig_imp_new_11-1/Gk3_imp_new_11
    # plt.plot(fermion_om,Delta3_11[n:2*n].imag,label='Delta3_11 imag')

    #use cluster DMFT
    # Delta4_11=iom+mu-sig_imp_new_11-Gk_imp_new_22/(Gk_imp_new_11*Gk_imp_new_22-Gk_imp_new_12**2)
    # plt.plot(Delta4_11[n:2*n].imag,label='cluster DMFT')

    # analytical limit
    # eps2=calc_eps2_ave(knum)
    # plt.plot(-1*(eps2/fermion_om).real,label='high freq limit')

    # delta without perturbation.
    # plt.plot(Delta0_11[n:2*n].imag,label='DMFT')

    #generate delta from HT
    # x, Di = np.loadtxt('DOS_3D.dat').T
    # W = hilbert.Hilb(x,Di)
    om = (2*np.arange(n)+1)*np.pi/beta
    # Dlt_A,Dlt_B = hilbert.SCC_AFM(W, om, beta, mu, U, sigA, sigB, False)
    # plt.plot(om,Dlt_A.real,label='DeltaHT_11 real')
    # plt.plot(om,Dlt_A.imag,label='DeltaHT_11 imag')
    # plt.plot(om,Dlt_B.real,label='DeltaHT_22 real')
    # plt.plot(om,Dlt_B.imag,label='DeltaHT_22 imag')
    # plt.xlabel("index of matsubara freqs")
    # plt.ylabel("Delta.imag")
    # plt.legend()
    # plt.grid()
    # plt.show()
    f = open(fileD, 'w')
    for i,iom in enumerate(om):
        print(iom, Delta_11[i+n].real, Delta_11[i+n].imag, Delta_22[i+n].real, Delta_22[i+n].imag, file=f) 
    f.close()
    return 0#Delta_11[n:2*n],Delta_22[n:2*n]


#when call functions in this file, comment everything below!
fileS = 'Sig.OCA'
fileD= 'Delta.inp'
knum=10 # default

# collect command line parameters
if (len(sys.argv)>=3):
    U=float(sys.argv[1])
    T=float(sys.argv[2])
if (len(sys.argv)>=4):
    fileS=sys.argv[3]
if (len(sys.argv)>=5):
    fileD=sys.argv[4]
if (len(sys.argv)>=6):
    knum=int(sys.argv[5])
if (len(sys.argv)>=7) or (len(sys.argv)<3):
    if rank ==0:
        print('input format does not match!\n format: mpirun -np 8 python perturb_mpi.py U T sigfile deltafile knum\nsigfile deltafile knum are optional')
        print('example: mpirun -np 8 python perturb_mpi.py 7.0 0.38 Sig.dat')

if rank==0:
    print('-----------Perturbed Iteration of DMFT------')
    print('T=',T,'U=',U,'knum=',knum,'sigfile=',fileS,'deltafile=',fileD)

if (os.path.exists(fileS)):
    Sf = np.loadtxt(fileS).T
    sigA = Sf[1,:]+Sf[2,:]*1j
    sigB = Sf[3,:]+Sf[4,:]*1j
else:
    if rank==0:
        print('cannot find {}!'.format(fileS))


# sym_mapping(1,2,3)
# calc_sym_array(10)
# G_test(sigA,sigB,U,T,knum)
# Delta_DMFT(sigA,sigB,U,T,knum)
# precalcP_test(sigA,sigB,U,T,knum)
# sig_imp_pert_test(sigA,sigB,U,T,knum)
# new_sig(sigA,sigB,U,T,knum)
Delta_pert_DMFT(sigA,sigB,U,T,knum)
