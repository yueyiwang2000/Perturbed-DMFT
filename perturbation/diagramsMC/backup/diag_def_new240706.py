from scipy import *
from scipy.interpolate import interp1d
# import weight_lib 
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

def gen_tau_limit(depend,Ndimk,Ndimtau,ngf):
    '''
    This is function is used to generate correct limit for GFs at tau=0.
    Here an important issue is to pick correct limit, 0+ or 0-.
            #usually, each interaction has a tau index so all GFs should has the tau dependence like :G(tau_i-tau_j).
            # my definition is, if tau_i==tau_j, take 0+ if if i<j, take 0- if i>j. 
    The input is the labeling or dependence of GF.
    '''
    taulimit=np.ones(ngf,dtype=int)*2# initialize it to 2. 2=unspecified 1=0- 0=0+
    for i in np.arange(ngf):
        for j in np.arange(Ndimk,Ndimk+Ndimtau):
            if depend[i,j]==1:
                taulimit[i]=0# take 0+
                break
            if depend[i,j]==-1:
                taulimit[i]=1#take 0-
                break
    return taulimit

class FuncNDiagNew:
    """  
    This class is trying to define a class which suites different order of diagrams. And it starts from Sigma diagrams but not Phi.
    functions:
    init: initialization.
    call: (re)evaluate the integrand from calculated component of GFs.
    update: update some of GFs when the configuration get changed.


    For the result we should reserve some space for the trial before it is accepted. 
    Before accepting by metropolis, we should always keep the old results.

    initialization parameters:
    T: temperature
    U: Hubbard U
    knum: k space grid: knum*knum*knum in 3D space.
    nfreq: number of matsubara frequencies
    Ndimk: number of k variables. In a nth order phi diagram we need n+1 k indices to label the diagram.
    Ndimtau: number of tau variables. This should be norder-1 for sigma diagrams.
    norder: order of the diagram

    GFs: usually a tuple of (G11,G12,G22), the non-interacted propagaters used to build diagrams.

    ut: svd basis. has the shape (lmax,taunum)

    SubLatInd: SublatticeIndices of each point in Phidiagram. 1= site1 2=site2
        example: permutation of 2nd order Phi diagram should be [2,3,0,1]
        and the sublattice could be [1,2,1,2] for points [0,1,2,3]
        Remember interaction U connects 2 points with different spins so actually 1up=2dn.
        Later this will be removed since sublattice indices are included in configuration space.

    dependence: a matrix which tells which GF depend on which variable. 
        example: diagonal 2nd order should be: [[1,1,0,1]    1st GF: k+q, tau 
                                                [0,0,1,1]    2nd GF: k',tau
                                                [0,1,0,-1]   3nd GF: k'+q,-tau
                                                [1,0,0,-1]    4th GF: k, -tau 
        if only k get changed we only reevaluate the 1st and 4th GF.
        the svd basis ut(tau) should only depend of the external tau. and its dependence not included here.
        In principle this dependence can be generated through a function of permutation representation of Phi diagram but it is not done yet.
    
    
    """
    def __init__(self,T,U,knum,taunum,nfreq,norder,ut,perm,GFs,dependence,cut):
        self.norder=norder
        self.nGf=2*norder# number of GFs in a phi diagram.
        self.Ndimk = norder+1# number of k is always norder+1
        self.Ndimtau=norder-1# number of tau is always norder-1, for self-energy diagrams.
        self.knum=knum
        self.taunum=taunum
        self.nfreq=nfreq
        self.depend=dependence
        self.cut=cut
        self.taulimit=gen_tau_limit(self.depend,self.Ndimk,self.Ndimtau,self.nGf)
        self.U=U
        self.beta=1/T
        self.t=np.zeros(self.Ndimtau,dtype=int)
        self.k=np.zeros((self.Ndimk,3),dtype=int)
        self.ttemp=np.zeros(self.Ndimtau,dtype=int)
        self.ktemp=np.zeros((self.Ndimk,3),dtype=int)
        self.ksym_array=np.zeros(self.nGf,dtype=int)# the k-space symmetry: 0 means diagonal, 1 means off-diagonal.
        self.ut=ut#svd basis


        ori_grid=(np.arange(nfreq*2)+0.5)/(nfreq*2)
        
        self.slicedG=np.zeros(self.nGf)# the Gfs at current configuration.
        self.slicedG_temp=np.zeros(self.nGf)# the Gfs at current configuration. but this is for the trial of metropolis.
        self.G=np.zeros((self.nGf,taunum+1,knum,knum,knum))# later there might be one more dimension since we have more than 1 
        simp_grid=np.arange(taunum+1)/taunum
        # we do splining for all G11 G12 G22 first. then even we change sublatind we dont have to spline again.
        self.interpolator11=interp1d(ori_grid, GFs[0], kind='linear', axis=0, fill_value='extrapolate')
        self.interpolator12=interp1d(ori_grid, GFs[1], kind='linear', axis=0, fill_value='extrapolate')
        self.interpolator22=interp1d(ori_grid, GFs[2], kind='linear', axis=0, fill_value='extrapolate')
        self.SubLatInd=np.array([1,2,1,2,1,2])# later it will be a part of config space. Remember interaction connects 2 points with different spins!
        for i in np.arange(self.nGf):
            if self.SubLatInd[i]!=self.SubLatInd[perm[i]]:
                self.G[i,:,:,:,:]=self.interpolator12(simp_grid)
                self.ksym_array[i]=1
                # print('G{} is 12, sym={}'.format(i,self.ksym_array[i]))
            elif self.SubLatInd[i]==1 and self.SubLatInd[perm[i]]==1:
                self.G[i,:,:,:,:]=self.interpolator11(simp_grid)
                self.ksym_array[i]=0
                # print('G{} is 11, sym={}'.format(i,self.ksym_array[i]))
            elif self.SubLatInd[i]==2 and self.SubLatInd[perm[i]]==2:
                self.G[i,:,:,:,:]=self.interpolator22(simp_grid)
                self.ksym_array[i]=0
                # print('G{} is 22, sym={}'.format(i,self.ksym_array[i]))
        # initialization of slicedG
        for i in np.arange(self.nGf):
            ki=np.sum(self.depend[i,:self.Ndimk,None]*self.k,axis=0)# note: k should be an array with 3 elements.
            taui=np.sum(self.depend[i,self.Ndimk:]*self.t)
            # print(np.shape(self.G[i]),ki,taui,self.ksym_array[i],self.knum,self.taunum,self.taulimit[i])
            self.slicedG[i]=Gshift(self.G[i],ki,taui,self.ksym_array[i],self.knum,self.taunum,self.taulimit[i])
            self.slicedG_temp[i]=self.slicedG[i]
        # print('taulimit=',self.taulimit)

    def update(self,new_momentum,new_imagtime,l,cut):
        '''
        the position tells which variables are updated. it should be an array with the size=(Ndimk+Ndimtau). if element=1 then update GFs related with this variable.
        cut: in permutation rep, which propagator is cut. this cut number means the starting point of the cut propagator.
        '''
        # which variable is changed?
        varlist=np.zeros(self.Ndimk+self.Ndimtau)
        update=np.zeros(self.nGf)
        for i in np.arange(self.Ndimk):
            if np.array_equal(self.k[i],new_momentum[i])==0:
                varlist[i]=1
                self.k[i]=new_momentum[i]
        for i in np.arange(self.Ndimtau):
            if self.t[i]!=new_imagtime[i]:
                varlist[i+self.Ndimk]=1
                self.t[i]=new_imagtime[i]
        # which GFs should be updated?
        for i in np.arange(self.nGf):# all Gfs
            for j in np.arange(self.nGf):# all updated variables
                if self.depend[i,j]!=0 and varlist[j]==1:# this GF depend on this updated variable,
                    update[i]=1# this Gf should be updated
        # update part of propagators
        # print('k=\n',self.k)
        # print('tau=\n',self.t)
        for i in np.nonzero(update)[0]:
            # print('i=',i,update)
            ki=np.sum(self.depend[i,:self.Ndimk,None]*new_momentum,axis=0)
            taui=np.sum(self.depend[i,self.Ndimk:]*new_imagtime.T)
            # print('k{}=\n'.format(i),ki)
            # print(self.depend[i,self.Ndimk:],new_imagtime.T)
            # print('tau{}='.format(i),taui)
            self.slicedG[i]=Gshift(self.G[i],ki,taui,self.ksym_array[i],self.knum,self.taunum,self.taulimit[i])

        res=1.
        for i in np.arange(self.nGf):
            if cut!=i:# if this propagator is not cut
                res=res*self.slicedG[i]
        res=res*self.ut[l,new_imagtime[0]]#svd basis
        if np.abs(res)>1:
            print('warning! f(X) is unexpectedly huge!',res,'\nconfig=',new_momentum,new_imagtime)
            for i in np.arange(self.nGf):
                print('G{}={}'.format(i,self.slicedG[i]))
        #note: beta/taunum means dtau.
        # Note2: here the power of dtau is for all internal variables. However, using svd trick we also integrate out external tau.
        # the reason of not count dtau for external tau is that svd basis u(tau) is normalized without dtau: \sum_i u^2_l(tau_i)=1. No dtau here.
        return res*(-1)**1*(-self.U)**(self.norder)/self.knum**(3*(self.Ndimk-1))*(self.beta/self.taunum)**(self.Ndimtau-1)#-1 in the beginning means 1 loop
    
    def update_temp(self,new_momentum,new_imagtime,l,cut):
        '''
        the position tells which variables are updated. it should be an array with the size=(Ndimk+Ndimtau). if element=1 then update GFs related with this variable.
        This is for the trial. the original result should be kept.
        '''
        self.slicedG_temp=self.slicedG
        varlist=np.zeros(self.Ndimk+self.Ndimtau)
        for i in np.arange(self.Ndimk):
            if np.array_equal(self.k[i],new_momentum[i])==0:
                varlist[i]=1
                self.ktemp[i]=new_momentum[i]
        for i in np.arange(self.Ndimtau):
            if self.t[i]!=new_imagtime[i]:
                varlist[i+self.Ndimk]=1
                self.ttemp[i]=new_imagtime[i]
        update=np.zeros(self.nGf)
        for i in np.arange(self.nGf):# all Gfs
            for j in np.arange(self.nGf):# all updated variables
                if self.depend[i,j]!=0 and varlist[j]==1:# this GF depend on this updated variable,
                    update[i]=1# this Gf should be updated
        refresh=0
        for i in np.nonzero(update)[0]:
            k=np.sum(self.depend[i,:self.Ndimk,None]*new_momentum,axis=0)
            tau=np.sum(self.depend[i,self.Ndimk:]*(new_imagtime.T))
            self.slicedG_temp[i]=Gshift(self.G[i],k,tau,self.ksym_array[i],self.knum,self.taunum,self.taulimit[i])
        # if new_momentum[1,0]==0 and new_momentum[1,1]==0 and new_momentum[1,2]==0 and new_imagtime[0,0]==0:
        #         refresh=1
                
        res=1.
        for i in np.arange(self.nGf):
            if i!=cut:
                res=res*self.slicedG_temp[i]
        res=res*self.ut[l,new_imagtime[0]]
        if np.abs(res)>1:
            print('warning! f(X) is unexpectedly huge!',res,'\nconfig=',new_momentum,new_imagtime)
            for i in np.arange(self.nGf):
                print('G{}={}'.format(i,self.slicedG_temp[i]))
        # if refresh==1:
        #     print('old',self.k,self.t,update)
        #     print(self.slicedG_temp[0],self.slicedG_temp[1],self.slicedG_temp[2],self.slicedG_temp[3],self.slicedG_temp[4],self.slicedG_temp[5])

        return res*(-1)**1*(-self.U)**(self.norder)/self.knum**(3*(self.Ndimk-1))*(self.beta/self.taunum)**(self.Ndimtau-1)#-1 in the beginning means 1 loop
    
    def metropolis_accept(self):
        self.slicedG=self.slicedG_temp
        self.t=self.ttemp
        self.k=self.ktemp
        return 0

    def __call__(self, momentum,imagtime,l,cut):# this call re-evaluate the integrand
        res=1.
        for i in np.arange(self.nGf):
            if i!=cut:
                res=res*self.slicedG[i]
        res=res*self.ut[l,imagtime[0]]
        if res>1e3:
            print('warning! f(X) is unexpectedly huge!',res,'\nconfig=',momentum,imagtime)
            for i in np.arange(self.nGf):
                print('G{}={}'.format(i,self.slicedG[i]))
        return res
    
