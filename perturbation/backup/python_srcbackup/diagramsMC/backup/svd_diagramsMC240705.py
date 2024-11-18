from scipy import *
from scipy.interpolate import interp1d
import svd_weight_lib
from numpy import linalg
from numpy import random
import sys
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import time
import diag_def
import diag_def_new
from mpi4py import MPI
# import svdwrap as svd  # type: ignore
sys.path.append('../')
import perturb_lib as lib
import fft_convolution as fft
from diagramsMC_lib import *
import svdlib


'''
This code is MC estimation of self-energy diagrams using the trick of svd.
This function can be directly called. and the evaluation of integrand is accelerated.
'''
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

    
class params:
    def __init__(self):
        self.Nitt = 5000000   # number of MC steps in a single proc
        self.Ncout = 200000    # how often to print
        self.Nwarm = 1000     # warmup steps
        self.tmeassure = 10   # how often to meassure
        self.V0norm = 4e-2    # starting V0
        self.recomputew = 5e4/self.tmeassure # how often to check if V0 is correct
        self.per_recompute = 7 # how often to recompute fm auxiliary measuring function

def get_gtau(U,T):
    beta=1/T
    mu=U/2
    name1='../../files_boldc/{}_{}/Sig.out'.format(U,T)
    filename1=readDMFT(name1)
    name2='../../files_ctqmc/{}_{}/Sig.out'.format(U,T)
    filename2=readDMFT(name2)
    # print(filename1)
    # print(filename2)
    if (os.path.exists(filename1)):
        filename=filename1
    elif (os.path.exists(filename2)):
        filename=filename2
        # print('reading DMFT data from {}'.format(filename))
    else:
        print('{} cannot be found!'.format(filename))  
        return 0  
    
    sigma=np.loadtxt(filename)[:nfreq,:]
    check=sigma[-1,1]
    om=sigma[:,0]
    # anyways real part of sigA will be greater.
    if check>U/2:
        sigA=sigma[:,1]+1j*sigma[:,2]
        sigB=U-sigma[:,1]+1j*sigma[:,2]
    else:
        sigB=sigma[:,1]+1j*sigma[:,2]
        sigA=U-sigma[:,1]+1j*sigma[:,2]
    Sigma11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    Sigma11+=lib.ext_sig(sigA)[:,None,None,None]
    Sigma22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    Sigma22+=lib.ext_sig(sigB)[:,None,None,None]
    Sigma12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    z_1=lib.z4D(beta,mu,Sigma11,knum,nfreq)#z-delta
    z_2=lib.z4D(beta,mu,Sigma22,knum,nfreq)#z+delta
    G11_iom,G12_iom=lib.G_iterative(knum,z_1,z_2,Sigma12)
    G22_iom=-G11_iom.conjugate()
    G11_tau=fft.fermion_fft_diagG(knum,G11_iom,beta,sigA,mu)# currently sigma12=0
    G12_tau=fft.fast_ft_fermion(G12_iom,beta)
    G22_tau=G11_tau[::-1] 
    Gloc11_tau=np.sum(G11_tau,axis=(1,2,3))[:,None,None,None]/knum**3*np.ones((knum,knum,knum))[None,:,:,:]
    Gloc22_tau=np.sum(G22_tau,axis=(1,2,3))[:,None,None,None]/knum**3*np.ones((knum,knum,knum))[None,:,:,:]
    # Gloc11=np.sum(G11_tau,axis=(1,2,3))/knum**3
    # Gloc22=np.sum(G22_tau,axis=(1,2,3))/knum**3
    return G11_tau.real,G12_tau.real,G22_tau.real,Gloc11_tau.real,Gloc22_tau.real

def IntegrateByMetropolis_svd(func, qx, p,seed,lmax,ut,ifprint=1):
    """ Integration by Metropolis:
          func(momentum)   -- function to integrate
          qx               -- mesh given by a user
          p                -- other parameters
        Output:
          Pval(qx)
    """

    #-------basic settings-----
    # time check
    time_trial=0
    time_evaluate=0
    time_accrej=0
    time_others=0


    time_begin=time.time()
    ifrecomp=1
    np.random.seed(seed)# use the given seed
    # random.seed(0)         # make sure that we always get the same sequence of steps. If parallel. they should have different seeds.
    knum=func.knum
    taunum=func.taunum
    Pnorm2 = np.zeros_like(qx)  # Final results V_physical is stored in Pval
    
    Pnorm = 0.0            # V_alternative is stored in Pnorm
    Pval_sum = 0.0         # this is widetilde{V_physical}
    Pnorm_sum = 0.0        # this is widetilde{V_alternative}
    V0norm = p.V0norm      # this is V0
    dk_hist = 1.0          # we are creating histogram by adding each configuration with weight 1.
    # note: here i have both k and tau as external variable.
    Ndimk = func.Ndimk       # dimensions of the problem
    Ndimtau=func.Ndimtau
    Pval=np.zeros_like(qx)
    inc_recompute = (p.per_recompute+0.52)/p.per_recompute # How often to self-consistently recompute
    # the wight functions g_i and h_{ij}.




    momentum=np.random.randint(low=0, high=knum, size=(Ndimk,3))
    imagtime=np.random.randint(low=0, high=taunum, size=(Ndimtau,1))
    l=np.random.randint(0,lmax)# generate the external variable
    itau=imagtime[0] #and, tau is not external variable any more. but this is used for index of ut
    iQ=momentum[0:]
    
    myweight = svd_weight_lib.meassureWeight(Ndimk, Ndimtau,knum,taunum,lmax)
    fQ = func.update(momentum,imagtime,l,func.cut), V0norm * myweight( momentum,imagtime ) # fQ=(f(X), V0*f_m(X))
    # print('starting with f=', fQ, '\nstarting momenta=', momentum,'\n starting time=',imagtime)

    Nmeassure = 0  # How many measurements we had?
    Nall_q, Nall_k, Nall_w, Nacc_q, Nacc_k = 0, 0, 0, 0, 0
    c_recompute = 0 # when to recompute the auxiliary function?
    for itt in range(p.Nitt):   # long loop
        time0=time.time()
        iloop = int( (Ndimk+Ndimtau+1) * random.rand() )   # which variable to change, iloop=0 changes external r_0
        accept = False
        if (iloop == 0):                      # changing external variable k
            Nall_q += 1                                      # how many steps changig external variable
            (K_new,  trialaccept) = TrialStep0_k(knum)
        elif (iloop > 0) and (iloop < Ndimk):# changing internal variable k
            Nall_k += 1
            (K_new,  trialaccept) = TrialStep1_k(iloop,momentum,knum)
        elif (iloop == Ndimk+Ndimtau): # changing external variable l
            # lnew=np.random.randint(0,lmax)
            # trial_ratio=1; trialaccept=1
            (lnew, trialaccept)=Trialstep0_l(lmax)
        else:# changing internal variable tau
            (tau_new, trialaccept)=TrialStep1_tau(iloop,imagtime,taunum,Ndimk)
        time1=time.time()
        time_trial+=(time1-time0)
        if (trialaccept): # trial step successful. We did not yet accept, just the trial step.
            if (iloop<Ndimk):# k is changed
                tmomentum= Give_new_K(momentum, K_new, iloop)
                timagtime=imagtime
                tl=np.copy(l)
            elif (iloop==Ndimk+Ndimtau):# l is changed
                tl=np.copy(lnew)
                timagtime=imagtime
                tmomentum=momentum
            else:# tau is changed
                # print('tau_new=',tau_new)
                timagtime=Give_new_tau(imagtime, tau_new, iloop,Ndimk)
                tmomentum=momentum
                tl=np.copy(l)
            time_beforecalc=time.time()

            fQ_new = func.update_temp(tmomentum,timagtime,tl,func.cut), V0norm * myweight(tmomentum,timagtime) # f_new
            # fQ_check=fun1(tmomentum,timagtime)*ut[tl,timagtime[0]]

            # print('fQ_new',func.slicedG_temp[0],func.slicedG_temp[1],func.slicedG_temp[2],func.slicedG_temp[3],func.slicedG_temp[5],'\n')
            # if np.abs(fQ_new[0]-fQ_check)/np.abs(fQ_check)>1e-5:
            #     print('diff is so large! t={} k={}, diff={}'.format(timagtime,tmomentum,np.abs(fQ_new[0]-fQ_check)/np.abs(fQ_check)))

            # if timagtime[0]<=1:
            #     print('l={} tau={}'.format(tl,timagtime),func(tmomentum,timagtime),ut[tl,timagtime[0]])
            time_aftercalc=time.time()
            time_evaluate+=(time_aftercalc-time_beforecalc)
            ratio = (abs(fQ_new[0])+fQ_new[1])/(abs(fQ[0])+fQ[1]) 
            accept = abs(ratio) > 1-random.rand() # Metropolis
            if accept: # the step succeeded
                func.metropolis_accept()
                if (iloop<Ndimk):
                    momentum[iloop] = K_new
                elif (iloop==Ndimk+Ndimtau):
                    l=np.copy(tl)
                else:
                    imagtime[iloop-Ndimk]=tau_new
                fQ = fQ_new
                if iloop==0 or  iloop==Ndimk+Ndimtau:#update external variable
                        Nacc_q += 1  # how many accepted steps of this type
                        iQ = tmomentum[0,:]     # the new external variable index
                        # itau=timagtime[0]
                else:
                        Nacc_k += 1  # how many accepted steps of this bin
        time2=time.time()
        time_accrej+=(time2-time1)
        if (itt >= p.Nwarm and itt % p.tmeassure==0 and trialaccept==1): # below is measuring every p.tmeassure stepsand trialaccept==1
            Nmeassure += 1   # new meassurements
            W = abs(fQ[0])+fQ[1]             # this is the weight we are using
            f0, f1 = fQ[0]/W, fQ[1]/W        # the two measuring quantities
            # Next line needs CORRECTION for homework 
            Pval[l,iQ[0],iQ[1],iQ[2]]  += f0                  # V_physical : integral up to a constant
            # if itt<=p.Nwarm+1000:
            #     print('Pnorm2[itau,iQ]=',np.shape(Pval[itau,iQ]),np.shape(Pval),iQ)            
            Pnorm2[l,iQ[0],iQ[1],iQ[2]]+=f1
            # if itt<=p.Nwarm+1000:
            #     print('Pnorm2[itau,iQ]=',Pnorm2[itau,iQ])     
            Pnorm     += f1                  # V_alternative : the normalization for the integral
            Pnorm_sum += f1                  # widetilde{V}_alternative, accumulated over all steps
            # Next line needs CORRECTION for homework 
            Wphs  = abs(f0)                  # widetilde{V}_{physical}, accumulated over all steps
            Pval_sum  += Wphs
            # if itt<=p.Nwarm+1000:
            #     print('f1, Pnorm, sum(Pnorm2)=',f1,Pnorm, np.sum(Pnorm2))
            # doing histogram of the simulation in terms of V_physical only.
            # While the probability for a configuration is proportional to f(X)+V0*fm(X), the histogram for
            # constructing g_i and h_{ij} is obtained from f(X) only. 
            # print('starting with f=', fQ, '\nstarting momenta=', momentum,'\n starting time=',imagtime)
            myweight.Add_to_K_histogram(dk_hist*Wphs, momentum,imagtime,l)

            
            if itt>10000 and itt % (p.recomputew*p.tmeassure) == 0:
                # Now we want to check if we should recompute g_i and h_{ij}
                # P_v_P is V_physical/V_alternative*0.1
                P_v_P = Pval_sum/Pnorm_sum * 0.1 
                # We expect V_physical/V_alternative*0.1=P_v_P to be of the order of 1.
                # We do not want to change V0 too much, only if P_V_P falls utside the
                # range [0.25,4], we should correct V0.
                change_V0 = 0
                if P_v_P < 0.25 and itt < 0.3*p.Nitt:  # But P_v_P above 0.25 is fine
                    change_V0 = -1  # V0 should be reduced
                    V0norm    /= 2  # V0 is reduced by factor 2
                    Pnorm     /= 2  # V_alternative is proportional to V0, hence needs to be reduced too. 
                    Pnorm_sum /= 2  # widetilde{V}_alternative also needs to be reduced
                    Pnorm2/=2
                if P_v_P > 4.0 and itt < 0.3*p.Nitt: # and P_v_P below 4 is also fine
                    change_V0 = 1   # V0 should be increased 
                    V0norm    *= 2  # actually increasing V0
                    Pnorm     *= 2
                    Pnorm_sum *= 2
                    Pnorm2*=2
                if change_V0:       # V0 was changed. Report that. 
                    schange = ["V0 reduced to ", "V0 increased to"]
                    if ifprint==1:
                        print('%9.2fM P_v_P=%10.6f' % (itt/1e6, P_v_P), schange[int( (change_V0+1)/2 )], V0norm )
                    # Here we decied to drop all prior measurements if V0 is changed.
                    # We could keep them, but the convergence can be better when we drop them.
                    Pval = np.zeros(np.shape(Pval))
                    Pnorm2=np.zeros(np.shape(Pval))
                    Pnorm = 0
                    Nmeasure = 0

                # about recomputing 
                # Next we should check if g_i and h_ij need to be recomputed.
                # This should not be done too often, and only in the first half of the sampling.
                if (c_recompute==0 and itt<0.5*p.Nitt and ifrecomp==1 ):
                    # At the beginning we recompute quite often, later not so often anymore
                    # as the per_recompute is increasing...
                    p.per_recompute = int(p.per_recompute*inc_recompute+0.5)
                    # We normalized f_m, hence all previous accumulated values are now of the order
                    # of 1/norm. We also normalize the new additions to histogram with similar value, 
                    # but 5-times larger than before.
                #Note: this normalization step does not matter what is the value from mtyweight.normalize_k_histogram.
                # the point is if we have many MC steps there might be some overflow issue
                # so we need to give a small factor in front of all current histograms.
                # also the future histograms should be with this factor.
                # here the factor 5 means later steps are more important since we have much better f_alter. so we give more weight to these steps.
                # # if everything is fine, dk_hist should not be so tiny and the two lines of code below will not be excecuted.
                    dk_hist *= 5*myweight.Normalize_K_histogram()
                    # if dk_hist < 1e-8: # Once dk becomes too small, just start accumulating with weight 1.
                    #     dk_hist = 1.0
                    myweight.recompute()# Here we actually recompute g_i and h_{ij}.
                    # print('after recomputing:momentum={}, imagtime={}'.format(momentum,imagtime))
                    # plt.plot(myweight.gx_tau[1].real,label='gxtau real')
                    # plt.plot(myweight.gx_tau[1].imag,label='gxtau imag')
                    # plt.legend()
                    # plt.show()
                    # fQ = func(momentum,imagtime)*ut[l,imagtime[0]], V0norm * myweight( momentum,imagtime ) # And now we must recompute V0*f_m, because f_m has changed!
                    fQ = func.update(momentum,imagtime,l,func.cut), V0norm * myweight(tmomentum,timagtime)
                    # print('myweight( momentum,imagtime )=',myweight( momentum,imagtime ))
                    # if ifprint==1:
                    print('{:9.2f}M recomputing rank={} f_m={} f_0={} dk_hist={}'.format(itt/1e6,rank, fQ[1],fQ[0],dk_hist))
                    # print('p.per_recompute=',p.per_recompute)
                c_recompute += 1
                # print('c_recompute',c_recompute)
                if c_recompute>=p.per_recompute : c_recompute = 0 # counting when we will recompute next.        



        if (itt+1)% p.Ncout == 0 : # This is just printing information
            P_v_P = Pval_sum/Pnorm_sum * 0.1 # what is curent P_v_P
            ratio = (abs(fQ_new[0])+fQ_new[1])/(abs(fQ[0])+fQ[1]) # current ratio
            if ifprint==1:
                print('step={}M, iQ={}, l={},f_new={}  {}, ratio={} P_V_P={}'.format((itt+1)/1e6,iQ,l,fQ_new[0],fQ_new[1], ratio,P_v_P))
            if ifprint==0 and rank==0:
                print('step={}M\t rank={}\t P_V_P={}'.format((itt+1)/1e6,rank,P_v_P))
            # print('step={}M, P_val={}  Pnorm={}, f_old={} {} P_V_P={}'.format((itt+1)/1e6,Pval,Pnorm,fQ_new[0],fQ_new[1], fQ[0], fQ[1],P_v_P))
        time3=time.time()
        time_others+=(time3-time2)
# have to fix the Markov chain and then it is useful to use something like size of qx to normalize.   
    # Pval2 =  Pval*V0norm / Pnorm2        
    Pval *=  (knum**3*lmax*V0norm / Pnorm) #  Finally, the integral is I = V0 *V_physical/V_alternative
    time_end=time.time()
    time_ttl=time_end-time_begin

    if ifprint==1:
        print('knum**3*taunum=',knum**3*taunum)
        print('Pnorm',Pnorm)
        print('Pnorm2 ave',np.mean(Pnorm2))
        print('Pnorm2 sum',np.sum(Pnorm2))
    print('time of rank {}: total={:.3f}s, trial={:.3f}s, evaluate={:.3f}s, accrej={:.3f}s, others={:.3f}s'.format(rank, time_ttl,time_trial,time_evaluate,time_accrej,time_others))
    return (Pval.real,myweight)


def Integratesvd_Parallel(func,qx,p,lmax,ut):
    seed = np.random.randint(0, 10000) + rank*29
    # seed=0
    (Pval,myweight)=IntegrateByMetropolis_svd(func, qx, p,seed,lmax,ut,0)# choose different seeds for different proc.
    Pval = np.ascontiguousarray(Pval)
    # comm.barrier()
    # print('rank={},Pval={}'.format(rank,Pval[0,0,0]))
    # print('shape of Pval=',np.shape(Pval))
    if rank==0:
        all_Pval=np.zeros((nprocs,lmax,func.knum,func.knum,func.knum),dtype=Pval.dtype)
        ave_Pval=np.zeros_like(Pval)
    else:
        all_Pval=None
        ave_Pval=None
    comm.Gather(Pval,all_Pval,root=0)
    if rank==0:
        ave_Pval=np.sum(all_Pval,axis=(0))/nprocs
    return ave_Pval


def MC_test(U,T,nfreq,knum):
    # print('test started')
    mu=U/2
    beta=1/T
    # taunum=nfreq*2
    taunum=200
    lmax=10# number of svd coefficient.
    G11_tau,G12_tau,G22_tau,Gloc11_tau,Gloc22_tau=get_gtau(U,T)

    if rank==0:
        print('U={}, T={},knum={}, taunum={}'.format(U,T, knum,taunum))
    p = params()

    # ----------the original way of generating sigma3----------
    # Poff=P12(G12_tau,knum,nfreq,U,beta)
    # Qoff=Q12(G12_tau,knum,nfreq)
    # sigma2=sig2(G11_tau,G12_tau,G22_tau,knum,nfreq,U,beta)
    # sigma2off=sig2offdiag(G11_tau,G12_tau,G22_tau,knum,nfreq,U,beta)
    # sigma2loc=sig2(Gloc11_tau,G12_tau,Gloc22_tau,knum,nfreq,U,beta)
    sigma3_1_111=sig3(G11_tau,G22_tau,knum,nfreq,U, beta)
    sigma3_2_111=sig3_2(G11_tau,G22_tau,knum,nfreq,U, beta)
    # sigma3loc=sig3(Gloc11_tau,Gloc22_tau,knum,nfreq,U, beta)
    # sigma3121=sig3_1_121(G11_tau,G12_tau,knum,nfreq,U, beta)
    # sigma3122=sig3_1_122(G11_tau,G12_tau,G22_tau,knum,nfreq,U, beta)
    fun1=diag_def.FuncNDiag3_1(T,U,knum, taunum,nfreq, 4, 2, G11_tau, G11_tau, G22_tau, G22_tau, G22_tau,0,0,0,0,0)#sigma3_1_111
    # fun1=diag_def.FuncNDiag3_2(T,U,knum, taunum,nfreq, 4, 2, G11_tau, G11_tau, G22_tau, G22_tau, G22_tau,0,0,0,0,0)#sigma3_2_111
    GFs=(G11_tau,G12_tau,G22_tau)

    sym=0# symmetry of the quantity. offdiag choose 1, diag choose 0
    # note about parallel svd: sigular matrix u and v are not unique, even when we run them in different procs, some components may be different,
    #at least up to a minus sign.
    # taulist=(np.arange(taunum)+0.5)/taunum*beta#
    taulist=(np.arange(taunum))/taunum*beta#
    omlist=(2*np.arange(2*nfreq)+1-2*nfreq)*np.pi/beta 
    ker=svdlib.fermi_kernel(taulist,omlist,beta)
    # only do svd in rank0, than broadcast
    ut=np.empty((lmax,taunum),dtype=float)
    if rank==0:
        ut,sig,v=svdlib.svd_kernel_fast(ker,lmax)
        # if diag this is all coeffs. if offdiag, we only need 1st, 3rd, 5th,...
        print('sigular values:',sig)
    ut = np.ascontiguousarray(ut[sym::(sym+1),:])
    comm.Bcast(ut, root=0)

    lmax=int(lmax/(sym+1))# if offdiag, we only need half of lmax. but if diag we need all of them.
    perm31=np.array([2,3,4,5,0,1])
    perm32=np.array([2,5,4,1,0,3])
    #                k,k',q1,q2,tau,tau1
    dep31=np.array([[1,0,1,0,0,1],
                    [0,1,-1,0,0,1],
                    [1,0,0,1,1,-1],
                    [0,1,0,-1,1,-1],
                    [1,0,0,0,-1,0],
                    [0,1,0,0,-1,0]])
    dep32=np.array([[1,0,1,0,0,1],
                    [0,1,0,0,1,0],
                    [1,0,0,1,1,-1],
                    [0,1,1,0,0,-1],
                    [1,0,0,0,-1,0],
                    [0,1,0,1,-1,1]])    
    # fun=diag_def_new.FuncNDiagNew(T,U,knum,taunum,nfreq,3,ut,perm31,GFs,dep31,4)# sig3_1_111
    fun=diag_def_new.FuncNDiagNew(T,U,knum,taunum,nfreq,3,ut,perm32,GFs,dep32,4)# sig3_2_111
    qx=np.zeros((lmax,knum,knum,knum),dtype=float)
    # qx=np.zeros(taunum)
    # (Pval_raw,myweight)=IntegrateByMetropolis_svd(fun, qx, p,0,lmax,ut,1)#serial test (func, qx, p,seed,lmax,ut,ifprint=1)
    Pval_raw=Integratesvd_Parallel(fun,qx,p,lmax,ut)#parallel test
    
    if rank==0:
        Pval=sym_ave(Pval_raw,knum,sym)# average of all k points with the same symmetry. offdiag choose 1, diag choose 0


        ori_grid=(np.arange(nfreq*2)+0.5)/(nfreq*2)*beta
        # simp_grid=(np.arange(taunum+1))/taunum
        simp_grid=taulist
        # this is to show the k-dependence of GF, but i group them into different symmetries.
        k_displayed=np.zeros((knum,knum,knum))
        symgroup=0

        # coeff_ave=np.sum(Pval,axis=(1,2,3))/knum**3
        # Gave=svdlib.restore_Gf(coeff_ave,ut)
        # print('coeff_ave',coeff_ave)
        # plt.plot(taulist,Gave.real,label='MC AVE restored') 
        # plt.plot((np.arange(2*nfreq)+0.5)/nfreq/2*beta,Gloc11_tau[:,0,0,0].real,label='original') 
        # plt.legend()
        # plt.show()
        for kx in np.arange(10):
            for ky in np.arange(10):
                for kz in np.arange(10):
                    if k_displayed[kx,ky,kz]==0:# if this point is not checked
                        all_sym_kpoints=lib.sym_mapping(kx,ky,kz,knum).tolist()# find all equivalent points
                        unique_set = set(tuple(x) for x in all_sym_kpoints)
                        all_unique_sym_kpoints = [list(x) for x in unique_set]# but there might be a lot of duplicates.
                        symgroup+=1# they all belongs to the same symgroup.
                        # print(all_unique_sym_kpoints)
                        for q in all_unique_sym_kpoints:
                            # about the original quantity
                            quant=sigma3_2_111[:,q[0],q[1],q[2]].real
                            # quant=sigma2loc[:,q[0],q[1],q[2]].real
                            # quant=sigma2off[:,q[0],q[1],q[2]].real
                            # quant=Gloc11_tau[:,q[0],q[1],q[2]].real
                            interpolator1 = interp1d(ori_grid, quant, kind='linear', fill_value='extrapolate')
                            G1=interpolator1(simp_grid)
                            coeff_BF=svdlib.inner_prod(G1,ut)
                            # print('coeff_BF=',coeff_BF)
                            restored_quantityBF=svdlib.restore_Gf(coeff_BF,ut)


                            coeffs=Pval[:,q[0],q[1],q[2]].real
                            print('coeffs for MC and BF{}: \n{}\n{}'.format(q,coeffs,coeff_BF))
                            # plt.plot(coeff_BF,label='BF')
                            # plt.plot(coeffs,label="MC")
                            # plt.xlabel('nth')
                            # plt.legend()
                            # plt.show()

                            restored_quantity=svdlib.restore_Gf(coeffs,ut)
                            plt.plot((np.arange(nfreq*2)+0.5)/(nfreq*2)*beta ,quant,label='BF')
                            plt.plot(taulist ,G1,label='BF splined')
                            plt.plot(taulist,restored_quantityBF,label='BF restored')
                            plt.plot(taulist,restored_quantity,label='MC1')
                            # plt.plot(restored_quantityBF,label='BF')
                            plt.legend()
                            plt.title('k=[{},{},{}], factor={}, symgroup={}'.format(q[0],q[1],q[2],q[3],symgroup))
                            plt.show()
                            k_displayed[q[0],q[1],q[2]]+=1
    return 0#Pval

if __name__ == "__main__":
    U=10.
    T=0.4
    knum=10
    nfreq=500
    MC_test(U,T,nfreq,knum)