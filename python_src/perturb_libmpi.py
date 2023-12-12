import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess,math
import time
import perturb_imp
from mpi4py import MPI
import fft_convolution
from perturb_lib import *
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()



def precalcP12_mpi(beta, knum, G1):
    '''
    prime=0: calculate P; =1: calculate P'. definition see fft_convolution.precalcPp._fft
    '''
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
    pointsperproc=math.ceil(max_sym_index/nprocs)
    # print(max_sym_index)
    partP = np.zeros((pointsperproc,2*n),dtype=np.float64)
    for qind in np.arange(qpoints[1]-qpoints[0]):
        if qind+pointsperproc*rank<max_sym_index:

            q=essential_kpoints[qind+pointsperproc*rank]
            # print('calculating P12.Process {} has qpoint:'.format(rank), q)
            partP[qind,:]=fft_convolution.precalcP_fft(q, knum, n, G1,beta,1)# convolution fast algorithm
            # partP[qind,:]-=precalcP12_innerloop(q, knum, n, G1)[::-1]/knum**3/beta# original slow algorithm
    gathered_P=np.zeros((nprocs,pointsperproc,2*n),dtype=np.float64)
    comm.Gather(partP,gathered_P,root=0)
    # gathered_P = comm.gather(partP, root=0)
    full_P=np.zeros((2*n, knum, knum, knum),dtype=float)
    if rank==0:
        # P is compacted for fast connection between procs. now unpack it:
        
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
    comm.Bcast(full_P, root=0)
    return full_P
    # else:
    #     return np.zeros_like(full_P)

def precalcP11_mpi(beta, knum, Gk, fullsig,mu):# here, we need original sigma.
    '''
    prime=0: calculate P; =1: calculate P'. definition see fft_convolution.precalcPp._fft
    '''
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
    pointsperproc=math.ceil(max_sym_index/nprocs)
    partP = np.zeros((pointsperproc,2*n),dtype=np.complex128)
    for qind in np.arange(qpoints[1]-qpoints[0]):
        if qind+pointsperproc*rank<max_sym_index:
            q=essential_kpoints[qind+pointsperproc*rank]
            # print('calculating P11.Process {} has qpoint:'.format(rank), q)
            # partP[qind,:]=precalcP11_innerloop(q, knum, n, Gk,Gk0,alphak,f_alphak,delta_inf,beta)
            #actually used one
            partP[qind,:]=fft_convolution.precalcP_fft_diag(q, knum, n, Gk,beta,delta_inf,alphak)
            # this also works, more direct but not so good.
            # partP[qind,:]=fft_convolution.precalcP_fft(q, knum, n, Gk,beta,0)
    gathered_P=np.zeros((nprocs,pointsperproc,2*n),dtype=np.complex128)
    comm.Gather(partP,gathered_P,root=0)
    # gathered_P = comm.gather(partP, root=0)
    full_P=np.zeros((2*n, knum, knum, knum),dtype=np.complex128)
    if rank==0:
        # P is compacted for fast connection between procs. now unpack it:
        
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
    comm.Bcast(full_P, root=0)
    return full_P
    # else:
    #     return np.zeros_like(full_P)


def precalcQ_mpi(beta, knum, Gk, fullsig,mu,opt):# here, we need original sigma.
    '''
    opt==1 means shift with sign!
    '''
    n = int(np.shape(Gk)[0] / 2)
    if knum % 2 != 0:
        print('knum should be a even number!')
        return 0

    # k1,k2,k3=gen_full_kgrids(knum)
    # delta_inf=np.abs(-mu+fullsig[-1].real)
    # alphak=np.sqrt(dispersion(k1,k2,k3)**2+delta_inf**2)

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
    pointsperproc=math.ceil(max_sym_index/nprocs)
    partP = np.zeros((pointsperproc,2*n),dtype=np.complex128)
    for qind in np.arange(qpoints[1]-qpoints[0]):
        if qind+pointsperproc*rank<max_sym_index:
            q=essential_kpoints[qind+pointsperproc*rank]
            # print('calculating P11.Process {} has qpoint:'.format(rank), q)
            # partP[qind,:]=precalcP11_innerloop(q, knum, n, Gk,Gk0,alphak,f_alphak,delta_inf,beta)
            #actually used one
            partP[qind,:]=fft_convolution.precalcQ_fft(q, knum, Gk,beta,opt)
            # this also works, more direct but not so good.
            # partP[qind,:]=fft_convolution.precalcP_fft(q, knum, n, Gk,beta,0)
    gathered_P=np.zeros((nprocs,pointsperproc,2*n),dtype=np.complex128)
    comm.Gather(partP,gathered_P,root=0)
    full_P=np.zeros((2*n, knum, knum, knum),dtype=np.complex128)
    if rank==0:
        # P is compacted for fast connection between procs. now unpack it:
        for proc in np.arange(nprocs):
            for ind in np.arange(pointsperproc):
                qind=proc*pointsperproc+ind
                if qind < max_sym_index:
                    q=essential_kpoints[qind]
                    full_P[:,q[0],q[1],q[2]]=gathered_P[proc,ind,:]
                    all_sym_kpoints=sym_mapping(q[0],q[1],q[2],knum)
                    for kpoint in all_sym_kpoints:
                        full_P[:,kpoint[0],kpoint[1],kpoint[2]]=full_P[:,q[0],q[1],q[2]]
        # restore freq domain sym
        # full_P[n+1:,:,:,:]=full_P[n-1::-1,:,:,:].conjugate()
    comm.Bcast(full_P, root=0)
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
    pointsperproc=math.ceil(max_sym_index/nprocs)
    partsig = np.zeros((pointsperproc,N),dtype=np.complex128)

    k1,k2,k3=gen_full_kgrids(knum)
    delta_inf=np.abs(-mu+fullsig[-1].real)
    alphak=np.sqrt(dispersion(k1,k2,k3)**2+delta_inf**2)

    for qind in np.arange(qpoints[1]-qpoints[0]):
        if qind+pointsperproc*rank<max_sym_index:
            q=essential_kpoints[qind+pointsperproc*rank]
            # print('calculating sig.Process {} has kpoint:'.format(rank), q)
            # partsig[qind,:]=precalcsig_innerloop(q, knum, n, Pk,Gk,opt)# slow algorithm
            if opt==11:#diagonal
                # partsig[qind,:]=fft_convolution.precalcsig_fft_diag(q,knum,Gk,Pk,beta,U,delta_inf,alphak)
                partsig[qind,:]=fft_convolution.precalcsig_fft(q,knum,Gk,Pk,beta,U,0)
            elif opt==12:#off-diagonal
                partsig[qind,:]=fft_convolution.precalcsig_fft(q,knum,Gk,Pk,beta,U,1)


    gathered_sig=np.zeros((nprocs,pointsperproc,N),dtype=np.complex128)
    comm.Gather(partsig,gathered_sig,root=0)
    full_sig=np.zeros((N, knum, knum, knum),dtype=np.complex128)
    if rank==0:
        # P is compacted for fast connection between procs. now unpack it:
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
    comm.Bcast(full_sig, root=0)
    return full_sig#*-1*U*U / beta * ( 1/ a / knum) ** 3#2*np.pi

def precalcsigp_mpi(U,beta, knum, Pk, Gk, opt,fullsig,mu,a=1):
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
    pointsperproc=math.ceil(max_sym_index/nprocs)
    partsig = np.zeros((pointsperproc,N),dtype=np.complex128)

    k1,k2,k3=gen_full_kgrids(knum)
    delta_inf=np.abs(-mu+fullsig[-1].real)
    alphak=np.sqrt(dispersion(k1,k2,k3)**2+delta_inf**2)

    for qind in np.arange(qpoints[1]-qpoints[0]):
        if qind+pointsperproc*rank<max_sym_index:
            q=essential_kpoints[qind+pointsperproc*rank]
            # print('calculating sig.Process {} has kpoint:'.format(rank), q)
            # partsig[qind,:]=precalcsig_innerloop(q, knum, n, Pk,Gk,opt)# slow algorithm
            if opt==11:#diagonal
                # partsig[qind,:]=fft_convolution.precalcsig_fft_diag(q,knum,Gk,Pk,beta,U,delta_inf,alphak)
                partsig[qind,:]=fft_convolution.precalcsigp_fft(q,knum,Gk,Pk,beta,U,0)
            elif opt==12:#off-diagonal
                partsig[qind,:]=fft_convolution.precalcsigp_fft(q,knum,Gk,Pk,beta,U,1)


    gathered_sig=np.zeros((nprocs,pointsperproc,N),dtype=np.complex128)
    comm.Gather(partsig,gathered_sig,root=0)
    full_sig=np.zeros((N, knum, knum, knum),dtype=np.complex128)
    if rank==0:
        # P is compacted for fast connection between procs. now unpack it:
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
    comm.Bcast(full_sig, root=0)
    return full_sig#*-1*U*U / beta * ( 1/ a / knum) ** 3#2*np.pi
    # else:
    #     return np.zeros_like(full_sig)


def precalcP11_mpi_simple(beta, knum, Gk):# here, we need original sigma.
    '''
    prime=0: calculate P; =1: calculate P'. definition see fft_convolution.precalcPp._fft
    '''
    n = int(np.shape(Gk)[0] / 2)
    if knum % 2 != 0:
        print('knum should be a even number!')
        return 0

    #preparation for this trick. should we only do these in 1 proc and bcast to other procs?
    k1,k2,k3=gen_full_kgrids(knum)
    #generate alpha, f(alpha) for freq sum
    # delta_inf=np.abs(-mu+fullsig[-1].real)
    # alphak=dispersion(kx,ky,kz)# another alpha for test.
    # alphak=np.sqrt(dispersion(k1,k2,k3)**2+delta_inf**2)
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
    pointsperproc=math.ceil(max_sym_index/nprocs)
    partP = np.zeros((pointsperproc,2*n),dtype=np.complex128)
    for qind in np.arange(qpoints[1]-qpoints[0]):
        if qind+pointsperproc*rank<max_sym_index:
            q=essential_kpoints[qind+pointsperproc*rank]
            # print('calculating P11.Process {} has qpoint:'.format(rank), q)
            # partP[qind,:]=precalcP11_innerloop(q, knum, n, Gk,Gk0,alphak,f_alphak,delta_inf,beta)
            #actually used one
            # partP[qind,:]=fft_convolution.precalcP_fft_diag(q, knum, n, Gk,beta,delta_inf,alphak)
            # this also works, more direct but not so good.
            partP[qind,:]=fft_convolution.precalcP_fft(q, knum, n, Gk,beta,0)
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


def precalcsig_mpi_simple(U,beta, knum, Pk, Gk, opt,a=1):
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
    pointsperproc=math.ceil(max_sym_index/nprocs)
    partsig = np.zeros((pointsperproc,N),dtype=np.complex128)

    k1,k2,k3=gen_full_kgrids(knum)

    for qind in np.arange(qpoints[1]-qpoints[0]):
        if qind+pointsperproc*rank<max_sym_index:
            q=essential_kpoints[qind+pointsperproc*rank]
            # print('calculating sig.Process {} has kpoint:'.format(rank), q)
            # partsig[qind,:]=precalcsig_innerloop(q, knum, n, Pk,Gk,opt)# slow algorithm
            if opt==11:#diagonal
                # partsig[qind,:]=fft_convolution.precalcsig_fft_diag(q,knum,Gk,Pk,beta,U,delta_inf,alphak)
                partsig[qind,:]=fft_convolution.precalcsig_fft(q,knum,Gk,Pk,beta,U,0)
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


def precalc_mpi(beta, knum, Gk,Pk, fullsig,mu,opt):# here, we need original sigma.
    '''
    a general function for this kind of calculation.... inprogress.
    opt==1 means shift with sign!
    '''
    n = int(np.shape(Gk)[0] / 2)
    if knum % 2 != 0:
        print('knum should be a even number!')
        return 0

    k1,k2,k3=gen_full_kgrids(knum)
    delta_inf=np.abs(-mu+fullsig[-1].real)
    alphak=np.sqrt(dispersion(k1,k2,k3)**2+delta_inf**2)

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
    pointsperproc=math.ceil(max_sym_index/nprocs)
    partP = np.zeros((pointsperproc,2*n),dtype=np.complex128)
    for qind in np.arange(qpoints[1]-qpoints[0]):
        if qind+pointsperproc*rank<max_sym_index:
            q=essential_kpoints[qind+pointsperproc*rank]
            # print('calculating P11.Process {} has qpoint:'.format(rank), q)
            # partP[qind,:]=precalcP11_innerloop(q, knum, n, Gk,Gk0,alphak,f_alphak,delta_inf,beta)
            #actually used one
            partP[qind,:]=fft_convolution.precalcQ_fft(q, knum, Gk,beta,opt)
            # this also works, more direct but not so good.
            # partP[qind,:]=fft_convolution.precalcP_fft(q, knum, n, Gk,beta,0)
    gathered_P=np.zeros((nprocs,pointsperproc,2*n),dtype=np.complex128)
    comm.Gather(partP,gathered_P,root=0)
    full_P=np.zeros((2*n, knum, knum, knum),dtype=np.complex128)
    if rank==0:
        # P is compacted for fast connection between procs. now unpack it:
        for proc in np.arange(nprocs):
            for ind in np.arange(pointsperproc):
                qind=proc*pointsperproc+ind
                if qind < max_sym_index:
                    q=essential_kpoints[qind]
                    full_P[:,q[0],q[1],q[2]]=gathered_P[proc,ind,:]
                    all_sym_kpoints=sym_mapping(q[0],q[1],q[2],knum)
                    for kpoint in all_sym_kpoints:
                        full_P[:,kpoint[0],kpoint[1],kpoint[2]]=full_P[:,q[0],q[1],q[2]]
        # restore freq domain sym
        # full_P[n+1:,:,:,:]=full_P[n-1::-1,:,:,:].conjugate()
    comm.Bcast(full_P, root=0)
    return full_P