'''
Note: This code can only be run in the Perturbed_DMFT/perturbation directory, as:
(mpirun -np 8) python -m diagramsMC.dispersive_phi.diagramsMC_closedPhi 
the parallel part in the beginning is recommended but optional.
This code is MC estimation of self-energy diagrams using the trick of svd.
This function can be directly called. and the evaluation of integrand is accelerated.
'''


from scipy import *
from scipy.interpolate import interp1d
from . import weight_lib_closedPhi#
from . import diag_def_closedPhi#
from numpy import linalg
from numpy import random
import sys
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import time
import diagrams,serial_module,perm_def

# import diag_def_cutPhi
# import diag_def_cutPhibackup
from mpi4py import MPI
sys.path.append('../')
sys.path.append('../../')
import perturb_lib as lib
import fft_convolution as fft
from ..diagramsMC_lib import *
# import basis
import copy



import pert_energy_lib


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()


class params:
    def __init__(self):
        self.Nitt = 5000000   # number of MC steps in a single proc
        self.Ncout = 1000000    # how often to print
        self.Nwarm = 1000     # warmup steps
        self.tmeassure = 10   # how often to meassure
        self.V0norm = 4e-2    # starting V0
        self.recomputew = 5e4/self.tmeassure # how often to check if V0 is correct
        self.per_recompute = 7 # how often to recompute fm auxiliary measuring function

def get_gtau(U,T):
    beta=1/T
    mu=U/2
    name1='../files_boldc/{}_{}/Sig.out'.format(U,T)
    filename1=readDMFT(name1)
    name2='../files_ctqmc/{}_{}/Sig.out'.format(U,T)
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
    return G11_tau.real,G12_tau.real,G22_tau.real,Gloc11_tau.real,Gloc22_tau.real,G11_iom,G12_iom,G22_iom

def IntegrateByMetropolis_phi(func, p,seed):
    """ Integration by Metropolis:
          func(momentum)   -- function to integrate
          qx               -- mesh given by a user
          p                -- other parameters
        Output:
          Pval(qx)
    """
    ifprint=0
    #-------basic settings-----
    # time check
    time_trial=0
    time_evaluate=0
    time_accrej=0
    time_others=0
    Nacc,Nrej=0,0
    time_begin=time.time()
    ifrecomp=1
    np.random.seed(seed)# use the given seed
    # random.seed(0)         # make sure that we always get the same sequence of steps. If parallel. they should have different seeds.
    knum=func.knum
    taunum=func.taunum
    taufold=np.arange(func.taunum+2)
    taufold[-1]=func.taunum-1
    taufold[func.taunum]=0
    kfold=np.arange(func.knum+2)
    kfold[-1]=func.knum-1
    kfold[func.knum]=0
    # Pnorm2 = np.zeros_like(qx)  # Final results V_physical is stored in Pval
    
    Pnorm = 0.0            # V_alternative is stored in Pnorm
    Pval_sum = 0.0         # this is widetilde{V_physical}
    Pnorm_sum = 0.0        # this is widetilde{V_alternative}
    V0norm = p.V0norm      # this is V0
    dk_hist = 1.0          # we are creating histogram by adding each configuration with weight 1.
    # note: here i have both k and tau as external variable.
    Ndimk = func.Ndimk       # dimensions of the problem
    Ndimtau=func.Ndimtau
    Ndimlat=func.Ndimlat
    Pval=0.
    inc_recompute = (p.per_recompute+0.52)/p.per_recompute # How often to self-consistently recompute
    # the wight functions g_i and h_{ij}.




    momentum=np.random.randint(low=0, high=knum, size=(Ndimk,3))
    imagtime=np.random.randint(low=0, high=taunum, size=(Ndimtau,1))
    sublatind=np.random.randint(low=1,high=3,size=(Ndimlat))# indices for each time point. 

    # if iflocal:
    #     sublatind=np.ones_like(sublatind)
    tmomentum=copy.deepcopy(momentum)
    timagtime=copy.deepcopy(imagtime)
    tsublatind=copy.deepcopy(sublatind)

    myweight = weight_lib_closedPhi.meassureWeight(Ndimk, Ndimtau,knum,taunum,Ndimlat)
    # to be updated. add sublatint in the update function.
    fQ = func.update(momentum,imagtime,sublatind), V0norm * myweight( momentum,imagtime,sublatind-1) # fQ=(f(X), V0*f_m(X)) sublatind consists 1 and 2 but we'd better start from 0.
    # print('starting with f=', fQ, '\nstarting momenta=', momentum,'\n starting time=',imagtime)

    Nmeassure = 0  # How many measurements we had?
    Nall_q, Nall_k, Nall_w, Nacc_q, Nacc_k = 0, 0, 0, 0, 0
    c_recompute = 0 # when to recompute the auxiliary function?
    for itt in range(p.Nitt):   # long loop
        time0=time.time()
        # variables: k,tau,sublatind, i_coeff,l_coeff,j_coeff
        iloop = int( (Ndimk+Ndimtau+Ndimlat) * random.rand() )   # which variable to change, iloop=0 changes external r_0
        accept = False
        if (iloop >= 0) and (iloop < Ndimk):# changing internal variable k
            Nall_k += 1
            (K_new,  trialaccept) = TrialStep1_k(iloop,momentum,knum,kfold)
            # if iflocal:
            #     trialaccept=0
        elif (iloop >= Ndimk) and (iloop < Ndimk+Ndimtau):# changing internal variable tau
            (tau_new, trialaccept)=TrialStep1_tau(iloop,imagtime,taunum,Ndimk,taufold)
        elif (iloop >= Ndimk+Ndimtau) and (iloop < Ndimk+Ndimtau+Ndimlat):# changing sublatint. does not matter in or external variable.
            sublatind_new=3-sublatind[iloop-Ndimk-Ndimtau]#np.random.randint(2)+1#
            trialaccept=1

            # if iflocal:
            #     trialaccept=0

        time1=time.time()
        time_trial+=(time1-time0)
        if (trialaccept): # trial step successful. We did not yet accept, just the trial step.
            if (iloop<Ndimk):# k is changed
                tmomentum= Give_new_K(momentum, K_new, iloop)
            elif (iloop<Ndimk+Ndimtau):# tau is changed
                timagtime=Give_new_tau(imagtime, tau_new, iloop,Ndimk)
            elif (iloop<Ndimk+Ndimtau+Ndimlat):
                tsublatind[iloop-Ndimk-Ndimtau]=sublatind_new 


            time_beforecalc=time.time()

            fQ_new = func.update_temp(iloop,tmomentum,timagtime,tsublatind), V0norm * myweight(tmomentum,timagtime,tsublatind-1) # f_new
            time_aftercalc=time.time()
            time_evaluate+=(time_aftercalc-time_beforecalc)
            ratio = (abs(fQ_new[0])+fQ_new[1])/(abs(fQ[0])+fQ[1]) 
            # print('ratio=',ratio)
            accept = abs(ratio) > 1-random.rand() # Metropolis
            if accept: # the step succeeded
                func.metropolis_accept(iloop)
                if (iloop<Ndimk):
                    momentum[iloop] = K_new
                elif iloop<Ndimk+Ndimtau:
                    imagtime[iloop-Ndimk]=tau_new
                elif iloop<Ndimk+Ndimtau+Ndimlat:
                    sublatind[iloop-Ndimk-Ndimtau]=tsublatind[iloop-Ndimk-Ndimtau]

                fQ = fQ_new
                Nacc+=1
                if iloop>=Ndimk+Ndimtau+Ndimlat:#update external variable: k, l, sublatind.
                        Nacc_q += 1  # how many accepted steps of this type
                else:
                        Nacc_k += 1  # how many accepted steps of this bin
            else:
                Nrej+=1
                time0=time.time()
                if (iloop<Ndimk):
                    tmomentum[iloop] = momentum[iloop]
                elif iloop<Ndimk+Ndimtau:
                    timagtime[iloop-Ndimk]=imagtime[iloop-Ndimk]
                elif iloop<Ndimk+Ndimtau+Ndimlat:
                    tsublatind[iloop-Ndimk-Ndimtau]=sublatind[iloop-Ndimk-Ndimtau]
                    # print('accept trialsublatind. new sublatind=',sublatind)

                    
                func.metropolis_reject(iloop)
                # print('metropolis rejected!\n')

                
        time2=time.time()
        time_accrej+=(time2-time1)
        if (itt >= p.Nwarm and itt % p.tmeassure==0 and trialaccept==1): # below is measuring every p.tmeassure stepsand trialaccept==1
            Nmeassure += 1   # new meassurements
            W = abs(fQ[0])+fQ[1]             # this is the weight we are using
            f0, f1 = fQ[0]/W, fQ[1]/W        # the two measuring quantities
            # Next line needs CORRECTION for homework 
            # if last sublatind=1, means sig11; if last sublatind=2. means sig12.
            Pval  += f0                  # V_physical : integral up to a constant
            # N[sublatind[0]-1]+=1
            # if itt<=p.Nwarm+1000:
            #     print('Pnorm2[itau,iQ]=',np.shape(Pval[itau,iQ]),np.shape(Pval),iQ)            
            # Pnorm2[l,iQ[0],iQ[1],iQ[2]]+=f1
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
            myweight.Add_to_K_histogram(dk_hist*Wphs, momentum,imagtime,sublatind-1)

            
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
                    # Pnorm2/=2
                if P_v_P > 4.0 and itt < 0.3*p.Nitt: # and P_v_P below 4 is also fine
                    change_V0 = 1   # V0 should be increased 
                    V0norm    *= 2  # actually increasing V0
                    Pnorm     *= 2
                    Pnorm_sum *= 2
                    # Pnorm2*=2
                if change_V0:       # V0 was changed. Report that. 
                    schange = ["V0 reduced to ", "V0 increased to"]
                    if ifprint==1:
                        print('%9.2fM P_v_P=%10.6f' % (itt/1e6, P_v_P), schange[int( (change_V0+1)/2 )], V0norm )
                    # Here we decied to drop all prior measurements if V0 is changed.
                    # We could keep them, but the convergence can be better when we drop them.
                    Pval = np.zeros(np.shape(Pval))
                    # N=np.zeros(2)
                    # Pnorm2=np.zeros(np.shape(Pval))
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
                    # dk_hist *= 5*myweight.Normalize_K_histogram()
                    # if dk_hist < 1e-8: # Once dk becomes too small, just start accumulating with weight 1.
                    #     dk_hist = 1.0
                    myweight.recompute()# Here we actually recompute g_i and h_{ij}.
                    # fQ = func(momentum,imagtime)*ut[l,imagtime[0]], V0norm * myweight( momentum,imagtime ) # And now we must recompute V0*f_m, because f_m has changed!
# update here!
                    fQ = func.update(momentum,imagtime,sublatind), V0norm * myweight(momentum,imagtime,sublatind-1)
                    # print('myweight( momentum,imagtime )=',myweight( momentum,imagtime ))
                    # if ifprint==1:
                    # print('{:9.2f}M recomputing rank={} f={} f_0={} dk_hist={} V0norm={}'.format(itt/1e6,rank, fQ[0],fQ[1],dk_hist,V0norm))
                    # print('p.per_recompute=',p.per_recompute)
                c_recompute += 1
                # print('c_recompute',c_recompute)
                if c_recompute>=p.per_recompute : c_recompute = 0 # counting when we will recompute next.        



        if (itt+1)% p.Ncout == 0 : # This is just printing information
            P_v_P = Pval_sum/Pnorm_sum * 0.1 # what is curent P_v_P
            ratio = (abs(fQ_new[0])+fQ_new[1])/(abs(fQ[0])+fQ[1]) # current ratio
            if ifprint==1:
                print('\nstep={}M, f_new={}  {}, ratio={} P_V_P={} '.format((itt+1)/1e6,fQ_new[0],fQ_new[1], ratio,P_v_P))
                # print('func.time_deepcopy,func.time_detectvar,func.time_detectGF,func.time_updateGF,func.time_sublatind,func.time_basis')
                # print(func.time_deepcopy,func.time_detectvar,func.time_detectGF,func.time_updateGF,func.time_sublatind,func.time_basis)
            if ifprint==0 and rank==0:
                print('step={}M\t rank={}\t  f_new={} {} P_V_P={} '.format((itt+1)/1e6,rank,fQ[0],fQ[1],P_v_P))

                # print('trial,    evaluate,    accrej,    others')
                # print(time_trial,time_evaluate,time_accrej,time_others)
                # print('k   tau   sublatind    basisk   basistau   basisind    test')
                # print(func.time_k,func.time_tau,func.time_sublatind,func.time_basist,func.time_basisk,func.time_basisind,func.time_test)

            # print('step={}M, P_val={}  Pnorm={}, f_old={} {} P_V_P={}'.format((itt+1)/1e6,Pval,Pnorm,fQ_new[0],fQ_new[1], fQ[0], fQ[1],P_v_P))
        time3=time.time()
        time_others+=(time3-time2)
# have to fix the Markov chain and then it is useful to use something like size of qx to normalize.   
    Pval *=  (V0norm / Pnorm) #  Finally, the integral is I = V0 *V_physical/V_alternative. *4 means we have 11 12 21 22.
    time_end=time.time()
    time_ttl=time_end-time_begin

    # if ifprint==1:
        # print('knum**3*taunum=',knum**3*taunum)
        # print('Pnorm',Pnorm)
        # print('Pnorm2 ave',np.mean(Pnorm2))
        # print('Pnorm2 sum',np.sum(Pnorm2))
    # print('time of rank {}: total={:.3f}s, trial={:.3f}s, evaluate={:.3f}s, accrej={:.3f}s, others={:.3f}s'.format(rank, time_ttl,time_trial,time_evaluate,time_accrej,time_others))
    return (Pval.real,myweight)


def Integratesvd_Parallel(func,p):
    seed = np.random.randint(0, 10000) + rank*29
    # seed=0
    (Pval,myweight)=IntegrateByMetropolis_phi(func, p,seed)# choose different seeds for different proc.
    Pval = np.ascontiguousarray(Pval)
    # comm.barrier()
    # print('rank={},Pval={}'.format(rank,Pval[0,0,0]))
    # print('shape of Pval=',np.shape(Pval))
    if rank==0:
        all_Pval=np.zeros((nprocs),dtype=Pval.dtype)
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
    taunum=100
    G11_tau,G12_tau,G22_tau,Gloc11_tau,Gloc22_tau,G11_iom,G12_iom,G22_iom=get_gtau(U,T)

    if rank==0:
        print('U={}, T={},knum={}, taunum={}'.format(U,T, knum,taunum))
    p = params()

    # ----------the original way of generating sigma3----------
    P22_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G22_tau,G22_tau,0)
    P12_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12,G12_tau,G12_tau,1)
    P11_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G11_tau,G11_tau,0)
    P11_iom=fft.fast_ift_boson(P11_tau,beta)
    P22_iom=fft.fast_ift_boson(P22_tau,beta)
    P12_iom=fft.fast_ift_boson(P12_tau,beta)
    Q11_tau=serial_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G22_tau,G11_tau,0)#Q=G_{s',-k}(tau)*G_{s,k+q}(tau)
    Q12_tau=serial_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,12, G12_tau,G12_tau,1)# Note: G12_-k=-G12_k!
    Q22_tau=serial_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G11_tau,G22_tau,0)
    Q11_iom=fft.fast_ift_boson(Q11_tau,beta)
    Q22_iom=fft.fast_ift_boson(Q22_tau,beta)
    Q12_iom=fft.fast_ift_boson(Q12_tau,beta)
    R11_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G22_tau,G11_tau,0)#R=G_{s',k}(-tau)*G_{s,k+q}(tau)
    R12_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12, G12_tau,G12_tau,1)
    R22_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G11_tau,G22_tau,0)
    R11_iom=fft.fast_ift_boson(R11_tau,beta)
    R22_iom=fft.fast_ift_boson(R22_tau,beta)
    R12_iom=fft.fast_ift_boson(R12_tau,beta)
    # Sig3_1_11,Sig3_1_12,Sig3_2_11,Sig3_2_12=diagrams.sig3(G11_iom,G12_iom,G11_tau,G12_tau,G22_tau,Q11_iom,Q12_iom,Q22_iom,R11_iom,R12_iom,R22_iom,knum,nfreq,U,beta)
    # Sig4_1_11,Sig4_1_12=diagrams.sig4_1(G11_tau,G12_tau,G22_tau,Q11_iom,Q12_iom,Q22_iom,knum,nfreq,U,beta)
    # Sig4_2_11,Sig4_2_12=diagrams.sig4_2(G11_tau,G12_tau,G22_tau,R11_iom,R12_iom,R22_iom,knum,nfreq,U,beta)
    Sig4_5_11,Sig4_5_12=diagrams.sig4_5(G11_tau,G12_tau,G22_tau,P11_iom,P12_iom,P22_iom,knum,nfreq,U,beta)

    om= (2*np.arange(nfreq)+1)*np.pi/beta

    # Phi from BF method
    Sig11=Sig4_5_11
    Sig12=Sig4_5_12
    Sig22=-Sig11.conjugate()
    s11_oo = Sig11[-1,:,:,:].real# currently this is a 3d array, each k point has a s_oo.
    EimpS11 = -mu+s11_oo # this is also a 3d array. G~1/(iom-eimp), so we need eimp.
    s22_oo = Sig22[-1,:,:,:].real
    EimpS22 = -mu+s22_oo
    # n11=(np.sum(G11_iom).real/knum**3/beta+1/2)
    # n22=(np.sum(G22_iom).real/knum**3/beta+1/2)
    
    # PhiBF=pert_energy_lib.fTrSigmaG(om, G11_iom[nfreq:], Sig11[nfreq:], EimpS11, beta,knum)+pert_energy_lib.fTrSigmaG(om, G22_iom[nfreq:], Sig22[nfreq:], EimpS22, beta,knum)+pert_energy_lib.fTrSigmaG_bf(om, G12_iom[nfreq:], Sig12[nfreq:], np.zeros((nfreq,knum,knum,knum)), beta,knum)*2
    # PhiBF+=np.sum(n11*s11_oo+n22*s22_oo)/knum**3 # remember to add the infinite part!

    Phi=pert_energy_lib.fTrSigmaG_bf(om, G11_iom[nfreq:,:,:,:], Sig11[nfreq:,:,:,:], EimpS11, beta,knum)*2#+pert_energy_lib.fTrSigmaG_bf(om, G22_iom[nfreq:,:,:,:], Sig22[nfreq:,:,:,:], EimpS22, beta,knum)
    Phi_offdiag=pert_energy_lib.fTrSigmaG_bf(om, G12_iom[nfreq:,:,:,:], Sig12[nfreq:,:,:,:], np.zeros((nfreq,knum,knum,knum)), beta,knum)*2
    PhiBF=Phi+Phi_offdiag


    GFs=(G11_tau,G12_tau,G22_tau)


    # some diagrams for testing
    # fun=diag_def_closedPhi.FuncNDiagNew(T,U,knum,taunum,nfreq,3,perm_def.perm31,GFs,perm_def.dep31)# sig3_1_111
    # fun=diag_def_closedPhi.FuncNDiagNew(T,U,knum,taunum,nfreq,3,perm_def.perm32,GFs,perm_def.dep32)# sig3_1_111
    # fun=diag_def_closedPhi.FuncNDiagNew(T,U,knum,taunum,nfreq,4,perm_def.perm41,GFs,perm_def.dep41)# sig4_2_111
    # fun=diag_def_closedPhi.FuncNDiagNew(T,U,knum,taunum,nfreq,4,perm_def.perm42,GFs,perm_def.dep42)# sig4_2_111
    fun=diag_def_closedPhi.FuncNDiagNew(T,U,knum,taunum,nfreq,4,perm_def.perm45,GFs,perm_def.dep45)# sig4_2_111
    # qx=np.zeros(taunum)
    # (Pval_raw,myweight)=IntegrateByMetropolis_svd(fun, qx, p,0,lmax,ut,1)#serial test (func, qx, p,seed,lmax,ut,ifprint=1)
    Pval_raw=Integratesvd_Parallel(fun,p)#parallel test
    
    if rank==0:
        print('Phi from MC:{}'.format(Pval_raw))
        print('Phi from BF:{}'.format(PhiBF))       
    return 0#Pval


def Summon_Integrate_Parallel_dispersive_Phi(func,p,symfac):
    F_n=0.
    F_n=Integratesvd_Parallel(func,p)# choose different seeds for different proc.
    F_n = comm.bcast(F_n, root=0)
    return F_n/symfac

if __name__ == "__main__":
    U=10.
    T=0.31
    knum=10
    nfreq=500
    MC_test(U,T,nfreq,knum)