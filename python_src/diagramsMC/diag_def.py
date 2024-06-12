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
sys.path.append('../')
import perturb_lib as lib
import fft_convolution as fft
from diagramsMC_lib import *


class FuncNDiag_P:
    def __init__(self,knum,taunum, nfreq,Ndimk,Ndimtau,G1tau,G2tau,opt1,opt2):
        self.Ndimk = Ndimk
        self.Ndimtau=Ndimtau
        self.knum=knum
        self.taunum=taunum
        self.nfreq=nfreq
        ori_grid=(np.arange(nfreq*2)+np.arange(1,nfreq*2+1))/2/(nfreq*2)# original imagtime grid. defined on [0,beta)
        # simp_grid=np.linspace(0,taunum,taunum+1)# originally we have too many tau points.
        simp_grid=np.arange(taunum+1)/taunum
        interpolator1 = interp1d(ori_grid, G1tau, kind='linear', axis=0, fill_value='extrapolate')
        interpolator2 = interp1d(ori_grid, G2tau, kind='linear', axis=0, fill_value='extrapolate')
        self.G1=interpolator1(simp_grid)
        self.G2=interpolator2(simp_grid)
        self.G1opt=opt1
        self.G2opt=opt2

    def __call__(self, momentum,imagtime):
        # this need some later adjustment. 
        k=momentum[1]#external
        q=momentum[0]
        tau=imagtime[0]#*self.tauinterval
        G1shift=Gshift(self.G1,k+q,tau,self.G1opt,self.knum,self.taunum)
        G2shift=Gshift(self.G2,k,-tau,self.G2opt,self.knum,self.taunum)
        res=(G1shift*G2shift).real/self.knum**3
        return res

class FuncNDiag_Q:
    def __init__(self,knum,taunum, nfreq,Ndimk,Ndimtau,G1tau,G2tau,opt1,opt2):
        self.Ndimk = Ndimk
        self.Ndimtau=Ndimtau
        self.knum=knum
        self.taunum=taunum
        self.nfreq=nfreq
        ori_grid=(np.arange(nfreq*2)+np.arange(1,nfreq*2+1))/2/(nfreq*2)# original imagtime grid. defined on [0,beta)
        # simp_grid=np.linspace(0,taunum,taunum+1)# originally we have too many tau points.
        simp_grid=np.arange(taunum+1)/taunum
        interpolator1 = interp1d(ori_grid, G1tau, kind='linear', axis=0, fill_value='extrapolate')
        interpolator2 = interp1d(ori_grid, G2tau, kind='linear', axis=0, fill_value='extrapolate')
        self.G1=interpolator1(simp_grid)
        self.G2=interpolator2(simp_grid)
        self.G1opt=opt1
        self.G2opt=opt2

    def __call__(self, momentum,imagtime):
        # this need some later adjustment. 
        k=momentum[1]#external
        q=momentum[0]
        tau=imagtime[0]#*self.tauinterval
        G1shift=Gshift(self.G1,k+q,tau,self.G1opt,self.knum,self.taunum)
        G2shift=Gshift(self.G2,-1*k,tau,self.G2opt,self.knum,self.taunum)
        res=(G1shift*G2shift).real/self.knum**3
        return res

class FuncNDiag_R:
    def __init__(self,knum,taunum, nfreq,Ndimk,Ndimtau,G1tau,G2tau,opt1,opt2):
        self.Ndimk = Ndimk
        self.Ndimtau=Ndimtau
        self.knum=knum
        self.taunum=taunum
        self.nfreq=nfreq
        ori_grid=(np.arange(nfreq*2)+np.arange(1,nfreq*2+1))/2/(nfreq*2)# original imagtime grid. defined on [0,beta)
        # simp_grid=np.linspace(0,taunum,taunum+1)# originally we have too many tau points.
        simp_grid=np.arange(taunum+1)/taunum
        interpolator1 = interp1d(ori_grid, G1tau, kind='linear', axis=0, fill_value='extrapolate')
        interpolator2 = interp1d(ori_grid, G2tau, kind='linear', axis=0, fill_value='extrapolate')
        self.G1=interpolator1(simp_grid)
        self.G2=interpolator2(simp_grid)
        self.G1opt=opt1
        self.G2opt=opt2

    def __call__(self, momentum,imagtime):
        # this need some later adjustment. 
        k=momentum[1]#external
        q=momentum[0]
        tau=imagtime[0]#*self.tauinterval
        G1shift=Gshift(self.G1,k+q,tau,self.G1opt,self.knum,self.taunum)
        G2shift=Gshift(self.G2,k,-tau,self.G2opt,self.knum,self.taunum)
        res=(G1shift*G2shift).real/self.knum**3
        return res

class FuncNDiag_order2:
    """  Gaussian through all diagonals k_i=k_j, i.e.:
    We have the following variables : k_0, k_1, k_2, ... k_{N-1}
    The function is
       fPQ = exp(k_{N-1}^2/width^2) * exp(|k_{N-2}-k_{N-1}|^2/width^2) * exp(-|k_{N-3}-k_{N-2}|^2/width^2) *...* exp(-|k_1-k_0|^2/width^2) * normalization
   We integrate only over k_1,k_2,...k_{N-1} and keep k_0 as external independent variable.
   We want the final result to be
      exp(-k_0^2/(width^2*N)) = Integrate[ fPQ d^3k_1 d^3k_2....d^3k_{N-1} ]
   Hence the normalization is
       normalization = N^(3/2) / ( sqrt(pi)*width )^(3(N-1))
    """
    def __init__(self,U,knum,taunum, nfreq,Ndimk,Ndimtau,G1tau,G2tau,G3tau,opt1,opt2,opt3):
        self.Ndimk = Ndimk
        self.Ndimtau=Ndimtau
        self.knum=knum
        self.taunum=taunum
        self.nfreq=nfreq
        ori_grid=(np.arange(nfreq*2)+np.arange(1,nfreq*2+1))/2/(nfreq*2)# original imagtime grid. defined on [0,beta)
        # simp_grid=np.linspace(0,taunum,taunum+1)# originally we have too many tau points.
        simp_grid=np.arange(taunum+1)/taunum
        interpolator1 = interp1d(ori_grid, G1tau, kind='linear', axis=0, fill_value='extrapolate')
        interpolator2 = interp1d(ori_grid, G2tau, kind='linear', axis=0, fill_value='extrapolate')
        interpolator3 = interp1d(ori_grid, G3tau, kind='linear', axis=0, fill_value='extrapolate')
        self.G1=interpolator1(simp_grid)
        self.G2=interpolator2(simp_grid)
        self.G3=interpolator3(simp_grid)
        self.G1opt=opt1
        self.G2opt=opt2
        self.G3opt=opt3   
        self.U=U

    def __call__(self, momentum,imagtime):
        # this need some later adjustment. 
        k=momentum[0]#external
        kp=momentum[1]
        q=momentum[2]
        tau=imagtime[0]#*self.tauinterval
        G1shift=Gshift(self.G1,k+q,tau,self.G1opt,self.knum,self.taunum)
        G2shift=Gshift(self.G2,kp,tau,self.G2opt,self.knum,self.taunum)
        G3shift=Gshift(self.G3,kp+q,-tau,self.G3opt,self.knum,self.taunum)
        res=-(G1shift*G2shift*G3shift).real*self.U**2/self.knum**6
        return res
    

class FuncNDiag:
    """  
    """
    def __init__(self,T,U,knum,taunum, nfreq,Ndimk,Ndimtau,G1tau,G2tau,G3tau,G4tau,G5tau,opt1,opt2,opt3,opt4,opt5):
        self.Ndimk = Ndimk
        self.Ndimtau=Ndimtau
        self.knum=knum
        self.taunum=taunum
        self.nfreq=nfreq
        ori_grid=(np.arange(nfreq*2)+np.arange(1,nfreq*2+1))/2/(nfreq*2)# original imagtime grid. defined on [0,beta)
        # simp_grid=np.linspace(0,taunum,taunum+1)# originally we have too many tau points.
        simp_grid=np.arange(taunum+1)/taunum
        interpolator1 = interp1d(ori_grid, G1tau, kind='linear', axis=0, fill_value='extrapolate')
        interpolator2 = interp1d(ori_grid, G2tau, kind='linear', axis=0, fill_value='extrapolate')
        interpolator3 = interp1d(ori_grid, G3tau, kind='linear', axis=0, fill_value='extrapolate')
        interpolator4 = interp1d(ori_grid, G4tau, kind='linear', axis=0, fill_value='extrapolate')
        interpolator5 = interp1d(ori_grid, G5tau, kind='linear', axis=0, fill_value='extrapolate')
        self.G1=interpolator1(simp_grid)
        self.G2=interpolator2(simp_grid)
        self.G3=interpolator3(simp_grid)
        self.G4=interpolator4(simp_grid)
        self.G5=interpolator5(simp_grid)

        self.G1opt=opt1
        self.G2opt=opt2
        self.G3opt=opt3
        self.G4opt=opt4
        self.G5opt=opt5     
        self.U=U
        self.beta=1/T

    def __call__(self, momentum,imagtime):
        # this need some later adjustment. 
        k=momentum[0]#external
        kp=momentum[1]
        q=momentum[2]
        qp=momentum[3]
        tau=imagtime[0]#*self.tauinterval
        tau1=imagtime[1]#*self.tauinterval
        G1shift=Gshift(self.G1,k+q,tau1,self.G1opt,self.knum,self.taunum)
        G2shift=Gshift(self.G2,k+qp,tau-tau1,self.G2opt,self.knum,self.taunum)
        G3shift=Gshift(self.G3,kp-q,tau1,self.G3opt,self.knum,self.taunum)
        G4shift=Gshift(self.G4,kp-qp,tau-tau1,self.G4opt,self.knum,self.taunum)
        G5shift=Gshift(self.G5,kp,-tau,self.G5opt,self.knum,self.taunum)
        res=(G1shift*G2shift*G3shift*G4shift*G5shift).real*self.U**3/self.knum**9*self.beta/self.taunum
        return res
