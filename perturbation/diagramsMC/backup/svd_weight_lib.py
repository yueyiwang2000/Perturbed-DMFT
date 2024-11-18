from scipy import *
from numpy import *
from numpy import linalg
from scipy import special
from numba import jit

'''
Now this code is tried to be modified to support the expanded configuration space which includes 
'''


@jit(nopython=True)
def self_f0(self_integral0):# return fm(r)/self_integral0
    #let's take the simplest form first..
    return 1./self_integral0

@jit(nopython=True)
def self_f1_k(momentum, ik, self_gxk):
    res=self_gxk[ik,momentum[ik,0],momentum[ik,1],momentum[ik,2]]
    # print('self_gxk=',self_gxk[ik,momentum[ik,0],momentum[ik,1],momentum[ik,2]])
    return max(res,1e-16)

# @jit(nopython=True)
def self_f1_tau(imagtime, itau, self_gxtau):
    res=self_gxtau[itau,imagtime[itau]]
    # print('self_gxtau',self_gxtau[itau,imagtime[itau]])
    return max(res,1e-16)

def self_f1_ind(sublatind, iind, self_gxind):
    res=self_gxind[iind,sublatind[iind]]
    return max(res,1e-16)

def self_Add_to_K_histogram(dk_hist, momentum, imagtime,l,sublatind,self_K_hist, self_tau_hist, self_l_hist,self_ind_hist,self_Ndimk,self_Ndimtau,self_Ndimind):

    #external histogram
    self_K_hist[0,momentum[0,0],momentum[0,1],momentum[0,2]]+=dk_hist
    # internal histogram.
    for ik in range(1,self_Ndimk):
        self_K_hist[ik,momentum[ik,0],momentum[ik,1],momentum[ik,2]]+=dk_hist
    self_tau_hist[0,imagtime[0]]+=dk_hist
    for itau in range(1,self_Ndimtau):
        self_tau_hist[itau,imagtime[itau]]+=dk_hist
    for iind in range(self_Ndimind):
        self_ind_hist[iind,sublatind[iind]]+=dk_hist
    self_l_hist[l]+=dk_hist
            
def self_fm(momentum,imagtime,sublatind, self_self_consistent, self_integral0,  self_gxk,self_gxtau,self_gxind):
    PQ_new = 1.0
    if not self_self_consistent:# in the very beginning, the trial.
        # for ik in range(1,len(momentum)):
        #     k = linalg.norm(momentum[ik])
        PQ_new *= self_f0(self_integral0 )
    else:
        # WHY we don't contain g_0?
        for ik in range(1,len(momentum)):
            # k = linalg.norm(momentum[ik])
            PQ_new *= self_f1_k( momentum, ik, self_gxk)
            # print('k:',ik,self_f1_k( momentum, ik, self_gxk))
        for itau in range(0,len(imagtime)):# here all taus are internal variable so from 0.
            PQ_new *= self_f1_tau( imagtime, itau, self_gxtau)
            # print('tau',itau,self_f1_tau( imagtime, itau, self_gxtau))
        for iind in range(0,len(sublatind)-1):#still, exclude external variable.
            PQ_new *= self_f1_ind(sublatind, iind, self_gxind)
    return PQ_new


class meassureWeight:
    """
       The operator() returns the value of a meassuring diagram, which is a function that we know is properly normalized to unity.
       We start with the flag self_consistent=0, in which case we use a simple function : 
                     f0(k) = theta(k<kF) + theta(k>kF) * (kF/k)^dexp
       Notice that f0(k) needs to be normalized so that \int f0(k) 4*pi*k^2 dk = 1.
    
       If given a histogram from previous MC data, and after call to Recompute(), it sets self_consistent=1, in which case we use
       separable approximation for the integrated function. Namely, if histogram(k) ~ h(k), then f1(k) ~ h(k)/k^2 for each momentum variable.
       We use linear interpolation for function g(k) and we normalize it so that \int g(k)*4*pi*k^2 dk = 1. 
       Notice that when performing the integral, g(k) should be linear function on the mesh i*dh, while the integral 4*pi*k^2*dk should be perform exactly.
    """
    def __init__(self,Ndimk, Ndimtau,knum,taunum,lmax,Ndimind):
        # self.Nbin = Nbin
        self.Nloops = Ndimk+Ndimtau+1
        self.Ndimk=Ndimk
        self.Ndimtau=Ndimtau
        self.knum=knum
        self.taunum=taunum
        self.lmax=lmax
        self.Ndimind=Ndimind
        # \int f0(k) d^3k, where f0(k) is given above
        self.integral0 =  knum**(3*(self.Ndimk-1))*taunum**(self.Ndimtau)*2**(self.Ndimind-1)# integration of trial fm, which is f0
        # at the beginning we do not have self-consistent function yet, but just f0
        self.self_consistent = False
        # self.Noff = Nloops-2 # how many off-diagonal h functions: 1 itself, 1 exterior variable
        # self.dh = cutoff/Nbin# width of each bin?
        # History of configurations
        self.K_hist = zeros((self.Ndimk,self.knum,self.knum,self.knum))
        self.tau_hist = zeros((self.Ndimtau,self.taunum))
        self.l_hist=zeros((self.lmax))
        self.ind_hist=zeros((self.Ndimind,2))
        self.gx_k     = zeros( (self.Ndimk,self.knum,self.knum,self.knum) )
        self.gx_tau     = zeros( (self.Ndimtau,self.taunum) )
        self.gx_l    = zeros( (self.lmax) )
        self.gx_ind=zeros((self.Ndimind,2))
    def __call__(self, momentum,imagtime,sublatind):
        return self_fm(momentum, imagtime,sublatind,self.self_consistent, self.integral0, self.gx_k,self.gx_tau,self.gx_ind)

    def Add_to_K_histogram(self, dk_hist, momentum, imagtime,l,sublatind):
        self_Add_to_K_histogram(dk_hist, momentum, imagtime,l,sublatind, self.K_hist,self.tau_hist,self.l_hist,self.ind_hist,self.Ndimk,self.Ndimtau,self.Ndimind)

    def Normalize_K_histogram(self):
        # We can get overflow during MPI for a long run. We should use a constant for normalization.
        # We could here normalize K_hist (with knorm), and than when we measure, we would
        # add instead of adding unity, we would add 1./knorm
        dnrm1 = self.Ndimk*(self.knum)**3/sum(self.K_hist)
        self.K_hist *= dnrm1
        dnrm2 = self.Ndimtau/sum(self.tau_hist)
        self.tau_hist *= dnrm2
        dnrm3=1/sum(self.l_hist)
        self.l_hist*=dnrm3
        dnrm=dnrm1*dnrm2*dnrm3
        # dnrm = (self.Ndimk+self.Ndimtau)/(sum(self.K_hist)+sum(self.tau_hist))
        print('normalize:norm=',dnrm)
        return dnrm
    
    def recompute(self):
        self.gx_k     = zeros( (self.Ndimk,self.knum,self.knum,self.knum) )
        self.gx_tau     = zeros( (self.Ndimtau,self.taunum) )
        self.gx_ind=zeros((self.Ndimind,2))
        self.self_consistent = True
        
        for itau in range(0,self.Ndimtau):
            self.gx_tau[itau,  :] =  self.tau_hist[itau,:]
            self.gx_tau[itau, 1:] += self.tau_hist[itau,:-1]
            self.gx_tau[itau,:-1] += self.tau_hist[itau,1:]
            self.gx_tau[itau,1:-1] *= 1./3.
            self.gx_tau[itau, 0]   *= 1/2.
            self.gx_tau[itau,-1]   *= 1/2.
        # actually gx_k also need smoothening... but let's skip it for now.    
        for ik in range(1,self.Ndimk):
            self.gx_k[ik,:,:,:]=self.K_hist[ik,:,:,:]
        for iind in range(0,self.Ndimind-1):
            self.gx_ind[iind,:]=self.ind_hist[iind,:]
        self.gx_l=self.l_hist


        for itau in range(0,self.Ndimtau):# normalization of gtau_i. now all taus are internal so start from 0.
            self.gx_tau[itau,:]*=1./sum(self.gx_tau[itau,:])#*self.beta/self.taunum
        for ik in range(1,self.Ndimk):
            self.gx_k[ik,:,:,:]*=1./sum(self.gx_k[ik,:,:,:])#/(self.knum)**3
        for iind in range(0,self.Ndimind-1):
            self.gx_ind[iind,:]*=1./sum(self.gx_ind[iind,:])
        print('recomputed gx_ind',self.gx_ind[0])
        self.gx_l*=1./sum(self.gx_l)
        # if the ansatz is as simple as vegas, I think that's it...?
        # fm=g1g2...gn, since gi's are all normalized, so as fm.