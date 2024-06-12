from scipy import *
from scipy.interpolate import interp1d
import weight_lib 
from numpy import linalg
from numpy import random
from scipy import special
import sys
import copy
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import time
import diag_def
from mpi4py import MPI
sys.path.append('../')
import perturb_lib as lib
import fft_convolution as fft
from diagramsMC_lib import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

    
class params:
    def __init__(self):
        self.Nitt = 5000000   # number of MC steps in a single proc
        self.Ncout = 50000    # how often to print
        self.Nwarm = 10000     # warmup steps
        self.tmeassure = 10   # how often to meassure
        self.V0norm = 4e-2    # starting V0
        self.recomputew = 5e4/self.tmeassure # how often to check if V0 is correct
        self.per_recompute = 7 # how often to recompute fm auxiliary measuring function


def IntegrateByMetropolis(func, qx, p,seed,ifprint=1):
    """ Integration by Metropolis:
          func(momentum)   -- function to integrate
          qx               -- mesh given by a user
          p                -- other parameters
        Output:
          Pval(qx)
    """
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
    iQ=momentum[0:]
    itau=imagtime[0]
# finish this measure function! this is just the first trial... should be simple in the beginning.
    myweight = meassureWeight(Ndimk, Ndimtau,knum,taunum)
    fQ = func(momentum,imagtime), V0norm * myweight( momentum,imagtime )  # fQ=(f(X), V0*f_m(X))
    # print('starting with f=', fQ, '\nstarting momenta=', momentum,'\n starting time=',imagtime)

    Nmeassure = 0  # How many measurements we had?
    Nall_q, Nall_k, Nall_w, Nacc_q, Nacc_k = 0, 0, 0, 0, 0
    c_recompute = 0 # when to recompute the auxiliary function?
    for itt in range(p.Nitt):   # long loop
        time0=time.time()
        iloop = int( (Ndimk+Ndimtau) * random.rand() )   # which variable to change, iloop=0 changes external r_0
        accept = False
        if (iloop == 0):                      # changing external variable k
            Nall_q += 1                                      # how many steps changig external variable
            (K_new, trial_ratio, trialaccept) = TrialStep0_k(knum)
        elif (iloop == Ndimk): # changing external variable tau
            (tau_new, trial_ratio, trialaccept)=TrialStep0_tau(taunum)
        elif (iloop > 0) and (iloop < Ndimk):# changing internal variable k
            Nall_k += 1
            (K_new, trial_ratio, trialaccept) = TrialStep1_k(iloop,momentum,knum)
        else:# changing internal variable tau
            (tau_new, trial_ratio, trialaccept)=TrialStep1_tau(iloop,imagtime,taunum,Ndimk)
        time1=time.time()
        time_trial+=(time1-time0)
        if (trialaccept): # trial step successful. We did not yet accept, just the trial step.
            if (iloop<Ndimk):
                tmomentum= Give_new_K(momentum, K_new, iloop)
                timagtime=imagtime
            else:
                # print('tau_new=',tau_new)
                timagtime=Give_new_tau(imagtime, tau_new, iloop,Ndimk)
                tmomentum=momentum
            time_beforecalc=time.time()
            fQ_new = func(tmomentum,timagtime), V0norm * myweight(tmomentum,timagtime) # f_new
            time_aftercalc=time.time()
            time_evaluate+=(time_aftercalc-time_beforecalc)
            ratio = (abs(fQ_new[0])+fQ_new[1])/(abs(fQ[0])+fQ[1]) * trial_ratio 
            accept = abs(ratio) > 1-random.rand() # Metropolis
            if accept: # the step succeeded
                if (iloop<Ndimk):
                    momentum[iloop] = K_new
                else:
                    imagtime[iloop-Ndimk]=tau_new
                fQ = fQ_new
                if iloop==0 or iloop == Ndimk:
                        Nacc_q += 1  # how many accepted steps of this type
                        iQ = tmomentum[0,:]     # the new external variable index
                        itau=timagtime[0]
                else:
                        Nacc_k += 1  # how many accepted steps of this type
        time2=time.time()
        time_accrej+=(time2-time1)
        if (itt >= p.Nwarm and itt % p.tmeassure==0 and trialaccept==1): # below is measuring every p.tmeassure stepsand trialaccept==1
            Nmeassure += 1   # new meassurements
            # Next line needs CORRECTION for homework 
            W = abs(fQ[0])+fQ[1]             # this is the weight we are using
            f0, f1 = fQ[0]/W, fQ[1]/W        # the two measuring quantities
            # Next line needs CORRECTION for homework 
            Pval[itau,iQ[0],iQ[1],iQ[2]]  += f0                  # V_physical : integral up to a constant
            # if itt<=p.Nwarm+1000:
            #     print('Pnorm2[itau,iQ]=',np.shape(Pval[itau,iQ]),np.shape(Pval),iQ)            
            Pnorm2[itau,iQ[0],iQ[1],iQ[2]]+=f1
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
            myweight.Add_to_K_histogram(dk_hist*Wphs, momentum,imagtime)

            
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
                    Pval = zeros(shape(Pval))
                    Pnorm2=zeros(shape(Pval))
                    Pnorm = 0
                    Nmeasure = 0

                # about recomputing 
                # Next we should check if g_i and h_ij need to be recomputed.
                # This should not be done too often, and only in the first half of the sampling.
                if (c_recompute==0 and itt<0.7*p.Nitt and ifrecomp==1 ):
                    # At the beginning we recompute quite often, later not so often anymore
                    # as the per_recompute is increasing...
                    p.per_recompute = int(p.per_recompute*inc_recompute+0.5)
                    # We normalized f_m, hence all previous accumulated values are now of the order
                    # of 1/norm. We also normalize the new additions to histogram with similar value, 
                    # but 5-times larger than before.
                    dk_hist *= 5*myweight.Normalize_K_histogram()
                    if dk_hist < 1e-8: # Once dk becomes too small, just start accumulating with weight 1.
                        dk_hist = 1.0
                    myweight.recompute()# Here we actually recompute g_i and h_{ij}.
                    # print('after recomputing:momentum={}, imagtime={}'.format(momentum,imagtime))
                    # plt.plot(myweight.gx_tau[1].real,label='gxtau real')
                    # plt.plot(myweight.gx_tau[1].imag,label='gxtau imag')
                    # plt.legend()
                    # plt.show()
                    fQ = func(momentum,imagtime), V0norm * myweight( momentum,imagtime ) # And now we must recompute V0*f_m, because f_m has changed!
                    # print('myweight( momentum,imagtime )=',myweight( momentum,imagtime ))
                    if ifprint==1:
                        print('{:9.2f}M recomputing f_m={} f_0={} dk_hist={}'.format(itt/1e6, fQ[1],fQ[0],dk_hist))
                    # print('p.per_recompute=',p.per_recompute)
                c_recompute += 1
                # print('c_recompute',c_recompute)
                if c_recompute>=p.per_recompute : c_recompute = 0 # counting when we will recompute next.        



        if (itt+1)% p.Ncout == 0 : # This is just printing information
            P_v_P = Pval_sum/Pnorm_sum * 0.1 # what is curent P_v_P
            ratio = (abs(fQ_new[0])+fQ_new[1])/(abs(fQ[0])+fQ[1]) # current ratio
            if ifprint==1:
                print('step={}M, iQ={}, itau={},f_new={}  {}, f_old={} {} P_V_P={}'.format((itt+1)/1e6,iQ,itau,fQ_new[0],fQ_new[1], fQ[0], fQ[1],P_v_P))
            if ifprint==0 and rank==0:
                print('step={}M\t rank={}\t P_V_P={}'.format((itt+1)/1e6,rank,P_v_P))
            # print('step={}M, P_val={}  Pnorm={}, f_old={} {} P_V_P={}'.format((itt+1)/1e6,Pval,Pnorm,fQ_new[0],fQ_new[1], fQ[0], fQ[1],P_v_P))
        time3=time.time()
        time_others+=(time3-time2)
# have to fix the Markov chain and then it is useful to use something like size of qx to normalize.   
    # Pval2 =  Pval*V0norm / Pnorm2        
    Pval *=  (knum**3*taunum*V0norm / Pnorm) #  Finally, the integral is I = V0 *V_physical/V_alternative
    time_end=time.time()
    time_ttl=time_end-time_begin

    if ifprint==1:
        print('knum**3*taunum=',knum**3*taunum)
        print('Pnorm',Pnorm)
        print('Pnorm2 ave',np.mean(Pnorm2))
        print('Pnorm2 sum',np.sum(Pnorm2))
    print('time of rank {}: total={:.3f}s, trial={:.3f}s, evaluate={:.3f}s, accrej={:.3f}s, others={:.3f}s'.format(rank, time_ttl,time_trial,time_evaluate,time_accrej,time_others))
    return (Pval,myweight)

def Integrate_Parallel(func,qx,p):
    seed = np.random.randint(0, 10000) + rank*13
    (Pval,myweight)=IntegrateByMetropolis(func, qx, p,seed,0)# choose different seeds for different proc. don't print things.
    if rank==0:
        all_Pval=np.zeros((nprocs,func.taunum,func.knum,func.knum,func.knum),dtype=Pval.dtype)
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
    taunum=int(nfreq/10)

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
    # the original way of generating sigma3
    
    # Poff=P12(G12_tau,knum,nfreq,U,beta)
    # Qoff=Q12(G12_tau,knum,nfreq)
    # sigma2=sig2(G11_tau,G12_tau,G22_tau,knum,nfreq,U,beta)
    # sigma2off=sig2offdiag(G11_tau,G12_tau,G22_tau,knum,nfreq,U,beta)
    # sigma2loc=sig2(Gloc11_tau,G12_tau,Gloc22_tau,knum,nfreq,U,beta)

    # sigma3=sig3(G11_tau,G22_tau,knum,nfreq,U, beta)
    # sigma3loc=sig3(Gloc11_tau,Gloc22_tau,knum,nfreq,U, beta)
    # sigma3121=sig3_1_121(G11_tau,G12_tau,knum,nfreq,U, beta)
    sigma3122=sig3_1_122(G11_tau,G12_tau,G22_tau,knum,nfreq,U, beta)


    # inputs for MC    
    Ndimk=4
    Ndimtau=2
    if rank==0:
        print('U={}, T={}, dimk={}, Ndimtau={}, knum={}, taunum={}'.format(U,T,Ndimk, Ndimtau, knum,taunum))
    p = params()
    # fun=diag_def.FuncNDiag_Q(knum, taunum,nfreq, Ndimk, Ndimtau, G12_tau, G12_tau,1,1)# Qoff
    # fun=diag_def.FuncNDiag_P(knum, taunum,nfreq, Ndimk, Ndimtau, G12_tau, G12_tau,1,1)# Poff
    # fun=diag_def.FuncNDiag_R(knum, taunum,nfreq, Ndimk, Ndimtau, G12_tau, G12_tau,1,1)# Roff

    # fun=diag_def.FuncNDiag_order2(U,knum, taunum,nfreq, Ndimk, Ndimtau, G12_tau, G12_tau, G12_tau,1,1,1)#sigma2off
    # fun=diag_def.FuncNDiag_order2(U,knum, taunum,nfreq, Ndimk, Ndimtau, G11_tau, G22_tau, G22_tau,0,0,0)#sigma2
    # fun=diag_def.FuncNDiag_order2(U,knum, taunum,nfreq, Ndimk, Ndimtau, Gloc11_tau, Gloc22_tau, Gloc22_tau,0,0,0)#sigma2loc

    # fun=diag_def.FuncNDiag(T,U,knum, taunum,nfreq, Ndimk, Ndimtau, G11_tau, G11_tau, G22_tau, G22_tau, G22_tau,0,0,0,0,0)#sigma3111
    # fun=diag_def.FuncNDiag(T,U,knum, taunum,nfreq, Ndimk, Ndimtau, G12_tau, G12_tau, G12_tau, G12_tau, G22_tau,1,1,1,1,0)#sigma3121
    fun=diag_def.FuncNDiag(T,U,knum, taunum,nfreq, Ndimk, Ndimtau, G12_tau, G22_tau, G12_tau, G11_tau, G12_tau,1,0,1,0,1)#sigma3122
    # fun=diag_def.FuncNDiag_simple3(U,knum, taunum,nfreq, Ndimk, Ndimtau, Gloc11, Gloc11, Gloc22, Gloc22, Gloc22)#sigma3loc


    qx=np.zeros((taunum,knum,knum,knum),dtype=complex)
    # qx=np.zeros(taunum)
    # (Pval_raw,myweight)=IntegrateByMetropolis(fun, qx, p,0,1)#serial test
    Pval_raw=Integrate_Parallel(fun,qx,p)#parallel test
    if rank==0:
        Pval=sym_ave(Pval_raw,knum,1)# average of all k points with the same symmetry. offdiag choose 1, diag choose 0




        # this is to show the k-dependence of GF, but i group them into different symmetries.
        k_displayed=np.zeros((knum,knum,knum))
        symgroup=0
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
                            plt.plot(sigma3122[:,q[0],q[1],q[2]].real,label='BF')
                            plt.plot(np.arange(taunum)*2*nfreq/taunum,Pval[:,q[0],q[1],q[2]].real,label='MC1')
                            plt.legend()
                            plt.title('k=[{},{},{}], factor={}, symgroup={}'.format(q[0],q[1],q[2],q[3],symgroup))
                            plt.show()
                            k_displayed[q[0],q[1],q[2]]+=1
    return 0

if __name__ == "__main__":
    U=4.
    T=0.17
    knum=10
    nfreq=500
    MC_test(U,T,nfreq,knum)
