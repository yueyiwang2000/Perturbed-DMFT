import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess
import time
import perturb_imp
from mpi4py import MPI
import fft_convolution
from perturb_lib import *
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
# update: pass p_tau instead of p(iOm)


def precalcP12_mpi(beta, knum, G1, a=1):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    n = int(np.shape(G1)[0] / 2)
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
    partP = np.zeros((pointsperproc,2*n),dtype=np.float64)
    for qind in np.arange(qpoints[1]-qpoints[0]):
        q=essential_kpoints[qind+pointsperproc*rank]
        # print('calculating P12.Process {} has qpoint:'.format(rank), q)
        
        partP[qind,:]=fft_convolution.precalcP_fft(q, knum, n, G1,beta,1)# convolution fast algorithm
        # partP[qind,:]-=precalcP12_innerloop(q, knum, n, G1)[::-1]/knum**3/beta# original slow algorithm

    gathered_P=np.zeros((nprocs,pointsperproc,2*n),dtype=np.float64)
    comm.Gather(partP,gathered_P,root=0)
    # gathered_P = comm.gather(partP, root=0)
    if rank==0:
        # P is compacted for fast connection between procs. now unpack it:
        full_P=np.zeros((2*n, knum, knum, knum),dtype=float)
        for proc in np.arange(nprocs):
            for ind in np.arange(pointsperproc):
                qind=proc*pointsperproc+ind
                if qind < max_sym_index:
                    q=essential_kpoints[qind]
                    full_P[:,q[0],q[1],q[2]]=gathered_P[proc,ind,:]
                    # restore k-space domain sym
                    all_sym_kpoints=sym_mapping(q[0],q[1],q[2],knum)
                    for kpoint in all_sym_kpoints:
                        full_P[:,kpoint[0],kpoint[1],kpoint[2]]=full_P[:,q[0],q[1],q[2]]*kpoint[3]
        # restore freq domain sym
        # full_P[n-1::-1,:,:,:]=full_P[n+1:,:,:,:]
        return full_P
    # else:
    #     MPI.Finalize()
    #     exit()

def precalcP11_mpi(beta, knum, Gk, fullsig,mu,a=1):# here, we need original sigma.
    n = int(np.shape(Gk)[0] / 2)
    if knum % 2 != 0:
        print('knum should be a even number!')
        return 0

    #preparation for this trick. should we only do these in 1 proc and bcast to other procs?
    k1,k2,k3=gen_full_kgrids(knum)
    #generate alpha, f(alpha) for freq sum
    delta_inf=np.abs(-mu+fullsig[-1].real)
    # alphak=dispersion(kx,ky,kz)# another alpha for test.
    alphak=np.sqrt(dispersion(k1,k2,k3)**2+delta_inf**2)
    #generate unperturbed Green's function

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
    partP = np.zeros((pointsperproc,2*n),dtype=np.complex128)
    for qind in np.arange(qpoints[1]-qpoints[0]):
        q=essential_kpoints[qind+pointsperproc*rank]
        # print('calculating P11.Process {} has qpoint:'.format(rank), q)
        # partP[qind,:]=precalcP11_innerloop(q, knum, n, Gk,Gk0,alphak,f_alphak,delta_inf,beta)
        partP[qind,:]=fft_convolution.precalcP_fft_diag(q, knum, n, Gk,beta,delta_inf,alphak)
    gathered_P=np.zeros((nprocs,pointsperproc,2*n),dtype=np.complex128)
    comm.Gather(partP,gathered_P,root=0)
    # gathered_P = comm.gather(partP, root=0)
    if rank==0:
        # P is compacted for fast connection between procs. now unpack it:
        full_P=np.zeros((2*n, knum, knum, knum),dtype=np.complex128)
        for proc in np.arange(nprocs):
            for ind in np.arange(pointsperproc):
                qind=proc*pointsperproc+ind
                if qind < max_sym_index:
                    q=essential_kpoints[qind]
                    full_P[:,q[0],q[1],q[2]]=gathered_P[proc,ind,:]
        # restore k-space domain sym
                    all_sym_kpoints=sym_mapping(q[0],q[1],q[2],knum)
                    for kpoint in all_sym_kpoints:
                        full_P[:,kpoint[0],kpoint[1],kpoint[2]]=full_P[:,q[0],q[1],q[2]]
        # restore freq domain sym
        # full_P[n+1:,:,:,:]=full_P[n-1::-1,:,:,:].conjugate()
        return full_P 

def precalcsig_mpi(U,beta, knum, Pk, Gk, opt,fullsig,mu,a=1):
    n = int(np.shape(Gk)[0]/2)
    N=2*n
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
    partsig = np.zeros((pointsperproc,N),dtype=np.complex128)

    k1,k2,k3=gen_full_kgrids(knum)
    delta_inf=np.abs(-mu+fullsig[-1].real)
    alphak=np.sqrt(dispersion(k1,k2,k3)**2+delta_inf**2)

    for qind in np.arange(qpoints[1]-qpoints[0]):
        q=essential_kpoints[qind+pointsperproc*rank]
        # print('calculating sig.Process {} has kpoint:'.format(rank), q)
        # partsig[qind,:]=precalcsig_innerloop(q, knum, n, Pk,Gk,opt)# slow algorithm
        if opt==11:#diagonal
            partsig[qind,:]=fft_convolution.precalcsig_fft_diag(q,knum,Gk,Pk,beta,U,delta_inf,alphak)
            # partsig[qind,:]=fft_convolution.precalcsig_fft(q,knum,Gk,Pk,beta,U,0)
        elif opt==12:#off-diagonal
            partsig[qind,:]=fft_convolution.precalcsig_fft(q,knum,Gk,Pk,beta,U,1)


    gathered_sig=np.zeros((nprocs,pointsperproc,N),dtype=np.complex128)
    comm.Gather(partsig,gathered_sig,root=0)
    if rank==0:
        # P is compacted for fast connection between procs. now unpack it:
        full_sig=np.zeros((N, knum, knum, knum),dtype=np.complex128)
        for proc in np.arange(nprocs):
            for ind in np.arange(pointsperproc):
                qind=proc*pointsperproc+ind
                if qind < max_sym_index:
                    q=essential_kpoints[qind]
                    full_sig[:,q[0],q[1],q[2]]=gathered_sig[proc,ind,:]
        # restore k-space domain sym
                    all_sym_kpoints=sym_mapping(q[0],q[1],q[2],knum)
                    for kpoint in all_sym_kpoints:
                        full_sig[:,kpoint[0],kpoint[1],kpoint[2]]=full_sig[:,q[0],q[1],q[2]]*(kpoint[3]**power)
        # restore freq domain sym
        # full_sig[N-1::-1,:,:,:]=full_sig[N:,:,:,:].conjugate()
        return full_sig#*-1*U*U / beta * ( 1/ a / knum) ** 3#2*np.pi


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
    if rank == 0:
        for kxind in np.arange(knum):
            for kyind in np.arange(knum):
                for kzind in np.arange(knum):
                    # plt.plot(P12[:,kxind,kyind,kzind].real,label='P12_real')
                    # plt.plot(P12[:,kxind,kyind,kzind].imag,label='P12_imag')
                    plt.plot(P11[:,kxind,kyind,kzind].real,label='P11_real')
                    plt.plot(P11[:,kxind,kyind,kzind].imag,label='P11_imag')
                    plt.legend()
                    plt.show()
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
    if rank ==0:
        start_time = time.time()
    mu=U/2
    beta=1/T
    n=sigA.size
    z_A=z(beta,mu,sigA)#z-delta
    z_B=z(beta,mu,sigB)#z+delta
    allsigA=ext_sig(beta,sigA)
    allsigB=ext_sig(beta,sigB)
    G11=G_11(knum,z_A,z_B)
    G12=G_12(knum,z_A,z_B)
    P12_tau=precalcP12_mpi(beta,knum,G12)
    P11_tau=precalcP11_mpi(beta,knum,G11,allsigA,mu)
    if rank !=0:
        P12_tau=np.zeros((2*n,knum,knum,knum),dtype=float)
        P11_tau=np.zeros((2*n,knum,knum,knum),dtype=complex)
    comm.Bcast(P12_tau, root=0)
    comm.Bcast(P11_tau, root=0)
    Boson_om = (2*np.arange(2*n+1)-2*n)*np.pi/beta 
    sig_11=precalcsig_mpi(U,beta,knum,P11_tau,G11,11,allsigA,mu)# actually P22 and G11. BUT P11=P22
    sig_new_12=precalcsig_mpi(U,beta,knum,P12_tau,G12,12,allsigA,mu)
    if rank ==0:
        sig_22=-sig_11.conjugate()
        # sig_pert_imp11,sig_pert_imp22=sig_imp_pert_test(sigA,sigB,U,T,knum)
        sig_pert_imp11=np.sum(sig_11,axis=(1,2,3))/knum**3
        sig_pert_imp22=np.sum(sig_22,axis=(1,2,3))/knum**3
        sig_new_11=allsigA[:, None, None, None]+sig_11-sig_pert_imp11[:, None, None, None]
        sig_new_22=allsigB[:, None, None, None]+sig_22-sig_pert_imp22[:, None, None, None]
        # FT_test(G11[2*n:3*n],knum)
        # FT_test(P11[n:2*n],knum)
        # FT_test(sig_11[n:2*n],knum)
        end_time=time.time()
        print('time=',end_time-start_time)
        # for kxind in np.arange(knum):
        #     for kyind in np.arange(knum):
        #         for kzind in np.arange(knum):
                    # plt.plot(sig_new_11[:, kxind, kyind, kzind].real,label='sig_new_11 real')
                    # plt.plot(sig_new_11[:, kxind, kyind, kzind].imag,label='sig_new_11 imag')
                    # plt.plot(sig_11[:, kxind, kyind, kzind].real,label='sig_11 real')
                    # plt.plot(sig_11[:, kxind, kyind, kzind].imag,label='sig_11 imag')
                    # plt.plot(allsigA.real,label='Sig_DMFT real')
                    # plt.plot(allsigA.imag,label='Sig_DMFT imag')     
                    # plt.plot(sig_new_12[:, kxind, kyind, kzind].real,label='sig_12 real')
                    # plt.plot(sig_new_12[:, kxind, kyind, kzind].imag,label='sig_12 imag')  
                    # plt.legend()
                    # plt.grid()
                    # plt.show()
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


if __name__ == "__main__":
    T=0.37
    U=7.0
    knum=10
    nfreq=500
    sigma=np.loadtxt('./trial_sigma/{}_{}.dat'.format(U,T))[:nfreq,:]
    sigA=sigma[:,1]+1j*sigma[:,2]#sig+delta
    sigB=sigma[:,3]+1j*sigma[:,4]#sig-delta

    # sym_mapping(1,2,3)
    # calc_sym_array(10)
    # G_test(sigA,sigB,U,T,knum)
    # Delta_DMFT(sigA,sigB,U,T,knum)
    # precalcP_test(sigA,sigB,U,T,knum)
    # sig_imp_pert_test(sigA,sigB,U,T,knum)
    new_sig(sigA,sigB,U,T,knum)
    # Delta_pert_DMFT(sigA,sigB,U,T,knum)
else:
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
