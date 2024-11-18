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

def gen_sublatint_full(short):
    order=np.shape(short)[0]
    full_sublatint=np.zeros(2*order,dtype=int)
    for i in np.arange(order):
        full_sublatint[2*i]=short[i]
        full_sublatint[2*i+1]=3-short[i]# 1 will be 2, 2 will be 1
    return full_sublatint


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


def gen_limit_factor(taulimit):
    count=np.zeros(2,dtype=int)
    taulimitfac=np.zeros_like(taulimit,dtype=float)
    for i,ele in enumerate(taulimit):
        count[ele]+=1
    allcounts=count[0]+count[1]
    # print(count)
    for i,ele in enumerate(taulimit):
        taulimitfac[i]=allcounts/count[ele]
    # print('taulimitfac=',taulimitfac)
    return taulimitfac



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
    def __init__(self,T,U,knum,taunum,nfreq,norder,ut,kbasis,sublatind_basis,perm,GFs,dependence,symfactor):
        # orders and dimensions
        self.norder=norder
        self.symfactor=symfactor
        self.nGf=2*norder# number of GFs in a phi diagram.
        self.Ndimk = norder+1# number of k is always norder+1
        self.Ndimtau=norder-1# number of tau is always norder, correspond to n interactions, 1 for translational symmetry.
        self.Ndimlat=norder# number of sublattice variable in config space. 
        #parameters
        self.knum=knum
        self.taunum=taunum
        self.nfreq=nfreq
        self.depend=dependence
        self.taulimit=gen_tau_limit(self.depend,self.Ndimk,self.Ndimtau,self.nGf)
        self.taulimitfactor=gen_limit_factor(self.taulimit)
        self.U=U
        self.beta=1/T
        #configuration space
        self.t=np.zeros(self.Ndimtau,dtype=int)
        self.k=np.zeros((self.Ndimk,3),dtype=int)
        self.ttemp=np.zeros(self.Ndimtau,dtype=int)
        self.ktemp=np.zeros((self.Ndimk,3),dtype=int)
        self.sublatint_short=np.ones(self.Ndimlat,dtype=int)
        self.SubLatInd=gen_sublatint_full(self.sublatint_short)# later it will be a part of config space. Remember interaction connects 2 points with different spins!
        self.SubLatInd_temp=self.SubLatInd
        self.ksym_array=np.zeros(self.nGf,dtype=int)# the k-space symmetry: 0 means diagonal, 1 means off-diagonal.
        self.perm=perm
        #basis
        self.ut=ut#svd basis
        self.kbasis=kbasis
        self.sublatbasis=sublatind_basis

        #initialization of GF
        ori_grid=(np.arange(nfreq*2)+0.5)/(nfreq*2)
        self.slicedG=np.zeros(self.nGf)# the Gfs at current configuration.
        self.slicedG_temp=np.zeros(self.nGf)# the Gfs at current configuration. but this is for the trial of metropolis.
        self.Gind=np.zeros(self.nGf)# only has 2,3, and 4. GF are quoted from the dictiionary below.
        self.Gind_temp=np.zeros(self.nGf)
        simp_grid=np.arange(taunum+1)/taunum
        # we do splining for all G11 G12 G22 first. then even we change sublatind we dont have to spline again.
        self.interpolator11=interp1d(ori_grid, GFs[0], kind='linear', axis=0, fill_value='extrapolate')
        self.interpolator12=interp1d(ori_grid, GFs[1], kind='linear', axis=0, fill_value='extrapolate')
        self.interpolator22=interp1d(ori_grid, GFs[2], kind='linear', axis=0, fill_value='extrapolate')
        G11=self.interpolator11(simp_grid)
        G12=self.interpolator12(simp_grid)
        G22=self.interpolator22(simp_grid)
        # make it easier to quote the interpolators. I made keys to be 2,3,4, because 1+1=2, 1+2=2+1=3, 2+2=4 so it is easier to call them when i need diagonal or off diagonal elements. 
        self.GF={
            2: G11,
            3: G12,
            4: G22,
        }
  

        for i in np.arange(self.nGf):# initialize Gind and ksymarray.
            self.Gind[i]=self.SubLatInd[i]+self.SubLatInd[perm[i]]
            self.ksym_array[i]=(self.SubLatInd[i]+self.SubLatInd[perm[i]])%2
        self.ksym_arraytemp=self.ksym_array

        # initialization of slicedG
        for i in np.arange(self.nGf):
            ki=np.sum(self.depend[i,:self.Ndimk,None]*self.k,axis=0)# note: k should be an array with 3 elements.
            taui=np.sum(self.depend[i,self.Ndimk:]*self.t)
            self.slicedG[i]=Gshift(self.GF[self.Gind[i]],ki,taui,self.ksym_array[i],self.knum,self.taunum,self.taulimit[i])


    def update(self,new_momentum,new_imagtime,newsublatindshort,i_coeff,l,j_coeff):
        '''
        the position tells which variables are updated. it should be an array with the size=(Ndimk+Ndimtau). if element=1 then update GFs related with this variable.
        cut: in permutation rep, which propagator is cut. this cut number means the starting point of the cut propagator.
        '''
        # which variable is changed? maybe this can be given from input
        # print('input newsublatindshort:',newsublatindshort)
        newsublatind=gen_sublatint_full(newsublatindshort)
        # print('initial sublatind:',newsublatind)
        varlist=np.zeros(self.Ndimk+self.Ndimtau)
        update=np.zeros(self.nGf)
        for i in np.arange(self.Ndimk):# if k updated, some GFs have to be updated
            if np.array_equal(self.k[i],new_momentum[i])==0:
                varlist[i]=1
                self.k[i]=new_momentum[i]
        for i in np.arange(self.Ndimtau):# if tau updated, some GFs have to be updated
            if self.t[i]!=new_imagtime[i]:
                varlist[i+self.Ndimk]=1
                self.t[i]=new_imagtime[i]
        
        # which GFs should be updated?
        for i in np.arange(self.nGf):# all Gfs
            for j in np.arange(self.nGf):# all updated variables
                if self.depend[i,j]!=0 and varlist[j]==1:# this GF depend on this updated variable,
                    update[i]=1# this Gf should be updated since k or tau indices are updated
        # print('change of sublatind',self.SubLatInd,newsublatind)
        for i,ele in enumerate(newsublatind):
            if ele!=self.SubLatInd[i]:
                update[i]=1
                pointbefore=np.where(self.perm==i)[0][0]
                update[pointbefore]=1# if this point is changed, 2 propagators connected to this point will be updated.
                #update GFs to be used. also the k space symmetry
                self.ksym_array[i]=(newsublatind[i]+newsublatind[self.perm[i]])%2
                self.ksym_array[pointbefore]=(newsublatind[i]+newsublatind[pointbefore])%2
                self.Gind[i]=newsublatind[i]+newsublatind[self.perm[i]]
                # print(np.where(self.perm==i))
                self.Gind[pointbefore]=newsublatind[i]+newsublatind[pointbefore]
                # print('these 2 propagators are changed:',i, pointbefore)
        self.SubLatInd=newsublatind
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
            self.slicedG[i]=Gshift(self.GF[self.Gind[i]],ki,taui,self.ksym_array[i],self.knum,self.taunum,self.taulimit[i])
        result=self.sum_allsigmas(self.slicedG,new_momentum,new_imagtime,newsublatind,i_coeff,l,j_coeff)

        #note: beta/taunum means dtau.
        # Note2: here the power of dtau is for all internal variables. However, using svd trick we also integrate out external tau.
        # the reason of not count dtau for external tau is that svd basis u(tau) is normalized without dtau: \sum_i u^2_l(tau_i)=1. No dtau here.
        # print('fQ:',self.slicedG[0],self.slicedG[1],self.slicedG[2],self.slicedG[3],self.slicedG[4],self.slicedG[5])
        return result*(-1)**1*(-self.U)**(self.norder)/self.knum**(3*(self.Ndimk-1))*(self.beta/self.taunum)**(self.Ndimtau-1)#-1 in the beginning means 1 loop
    
    def update_temp(self,new_momentum,new_imagtime,newsublatindshort,i_coeff,l,j_coeff):
        '''
        the position tells which variables are updated. it should be an array with the size=(Ndimk+Ndimtau). if element=1 then update GFs related with this variable.
        This is for the trial. the original result should be kept.
        '''
        self.slicedG_temp=copy.deepcopy(self.slicedG)
        # print('fQ before slicing:',self.slicedG[0],self.slicedG[1],self.slicedG[2],self.slicedG[3],self.slicedG[4],self.slicedG[5])
        self.SubLatInd_temp=copy.deepcopy(self.SubLatInd)
        self.Gind_temp=copy.deepcopy(self.Gind)
        newsublatind=gen_sublatint_full(newsublatindshort)
        # which variable is changed?
        varlist=np.zeros(self.Ndimk+self.Ndimtau)
        for i in np.arange(self.Ndimk):
            if np.array_equal(self.k[i],new_momentum[i])==0:
                varlist[i]=1
                self.ktemp[i]=new_momentum[i]
        for i in np.arange(self.Ndimtau):
            if self.t[i]!=new_imagtime[i]:
                varlist[i+self.Ndimk]=1
                self.ttemp[i]=new_imagtime[i]
        # which GFs should be updated?
        update=np.zeros(self.nGf)
        for i in np.arange(self.nGf):# all Gfs
            for j in np.arange(self.nGf):# all updated variables
                if self.depend[i,j]!=0 and varlist[j]==1:# this GF depend on this updated variable,
                    update[i]=1# this Gf should be updated
        # if only change k or tau, this block below is enough.
        # update part of propagators
        refresh=0
        for i in np.nonzero(update)[0]:
            k=np.sum(self.depend[i,:self.Ndimk,None]*new_momentum,axis=0)
            tau=np.sum(self.depend[i,self.Ndimk:]*(new_imagtime.T))
            self.slicedG_temp[i]=Gshift(self.GF[self.Gind[i]],k,tau,self.ksym_array[i],self.knum,self.taunum,self.taulimit[i])
        # if new_momentum[1,0]==0 and new_momentum[1,1]==0 and new_momentum[1,2]==0 and new_imagtime[0,0]==0:
        #     refresh=1
        #     print('new',self.k,self.t,update)
            # print('\nself.slicedG_temp[i]',self.slicedG_temp[i])
        self.ksym_arraytemp=copy.deepcopy(self.ksym_array)
        # in the case that we have to change sublatind. 
        # print('change of sublatind',self.SubLatInd,newsublatind)
        sublatcount=0
        for i,ele in enumerate(newsublatind):
            if ele!=self.SubLatInd[i]:
                # print('old sublatind:',self.SubLatInd,'new sublatind:',newsublatind)
                # print('GF #{} was {}, now is {}'.format(i,self.SubLatInd[i]+self.SubLatInd[self.perm[i]],newsublatind[i]+newsublatind[self.perm[i]]))

                sublatcount+=2
                # print('slicedG_{} before={}'.format(i,self.slicedG[i]))
                k=np.sum(self.depend[i,:self.Ndimk,None]*new_momentum,axis=0)
                tau=np.sum(self.depend[i,self.Ndimk:]*(new_imagtime.T))    
                self.ksym_arraytemp[i]=(newsublatind[i]+newsublatind[self.perm[i]])%2# k space sym for new propagator. 
                self.Gind_temp[i]=newsublatind[i]+newsublatind[self.perm[i]]
                self.slicedG_temp[i]=Gshift(self.GF[self.Gind_temp[i]],k,tau,self.ksym_arraytemp[i],self.knum,self.taunum,self.taulimit[i])
                # print('slicedG_{} after={}'.format(i,self.slicedG_temp[i]))
                # print(Gshift(self.GF[newsublatind[i]+newsublatind[self.perm[i]]],k,tau,self.ksym_arraytemp[i],self.knum,self.taunum,self.taulimit[i]))
                # print(Gshift(self.GF[self.SubLatInd[i]+self.SubLatInd[self.perm[i]]],k,tau,self.ksym_array[i],self.knum,self.taunum,self.taulimit[i]))

                pointbefore=np.where(self.perm==i)[0][0]
                # print('GF #{} was {}, now is {}'.format(pointbefore,self.SubLatInd[pointbefore]+self.SubLatInd[i],newsublatind[i]+newsublatind[pointbefore]))
                # print('slicedG_{} before={}'.format(pointbefore,self.slicedG[pointbefore]))
                k=np.sum(self.depend[pointbefore,:self.Ndimk,None]*new_momentum,axis=0)
                tau=np.sum(self.depend[pointbefore,self.Ndimk:]*(new_imagtime.T))    
                self.ksym_arraytemp[pointbefore]=(newsublatind[pointbefore]+newsublatind[i])%2# k space sym for new propagator. 
                self.Gind_temp[pointbefore]=newsublatind[i]+newsublatind[pointbefore]
                self.slicedG_temp[pointbefore]=Gshift(self.GF[self.Gind_temp[pointbefore]],k,tau,self.ksym_arraytemp[pointbefore],self.knum,self.taunum,self.taulimit[pointbefore])
                # print('slicedG_{} after={}'.format(pointbefore,self.slicedG_temp[pointbefore]))
                # print(Gshift(self.GF[newsublatind[i]+newsublatind[pointbefore]],k,tau,self.ksym_arraytemp[pointbefore],self.knum,self.taunum,self.taulimit[pointbefore]))
                # print(Gshift(self.GF[self.SubLatInd[i]+self.SubLatInd[pointbefore]],k,tau,self.ksym_array[pointbefore],self.knum,self.taunum,self.taulimit[pointbefore]))
                
        self.SubLatInd_temp=copy.deepcopy(newsublatind)
        if sublatcount!=0 and sublatcount!=2 and sublatcount!=4:
            print('sublatcount not correct!',self.SubLatInd,newsublatind)


        result=self.sum_allsigmas(self.slicedG_temp,new_momentum,new_imagtime,newsublatind,i_coeff,l,j_coeff)

        # if refresh==1:
        # print('fQ after slicing:',self.slicedG[0],self.slicedG[1],self.slicedG[2],self.slicedG[3],self.slicedG[4],self.slicedG[5])
        # print('fQtemp:',self.slicedG_temp[0],self.slicedG_temp[1],self.slicedG_temp[2],self.slicedG_temp[3],self.slicedG_temp[4],self.slicedG_temp[5])
        # print('final result=',result)
        return result*(-1)**1*(-self.U)**(self.norder)/self.knum**(3*(self.Ndimk-1))*(self.beta/self.taunum)**(self.Ndimtau-1)#-1 in the beginning means 1 loop
    
    def metropolis_accept(self):
        self.slicedG=copy.deepcopy(self.slicedG_temp)
        self.t=copy.deepcopy(self.ttemp)
        self.k=copy.deepcopy(self.ktemp)
        self.Gind=copy.deepcopy(self.Gind_temp)
        self.SubLatInd=copy.deepcopy(self.SubLatInd_temp)
        self.ksym_array=copy.deepcopy(self.ksym_arraytemp)

        return 0


    def sum_allsigmas(self,slices,momentum, imagtime, sublatind,i_coeff,l,j_coeff):
        '''
        Given all slicedG and dependnecies, sublatind, momentum, imagtime, symfactor
        sum over all 2*norder integrands from cutting propagators of Phi.
        i,l,j are indices of space, time, sublatind basis.

        Note: if cut a GF G^ab(k,tau) in diagram Phi, we'll get sigma^ba(k,-tau)=-sigma^ba(k,beta-tau)
        so we should use u_l(-tau), m_i(k) and n_j(b,a) to get the coefficient c_ijl.
        also, if the cut G should take tau=0- limit, sigma should take opposite limit, tau=0+, vice versa.
        '''
        mat=np.ones((self.nGf,self.nGf))*slices[None,:]
        np.fill_diagonal(mat, 1)
        # for i in np.arange(self.nGf):
        #     mat[i,i]=1 # do not count this propagator, in the prod function, i.e. cut it.
        res=np.prod(mat,axis=(1))
        # print('res=',res)

        # for i in np.arange(self.nGf):
        #     extk=np.sum(self.depend[i,:self.Ndimk,None]*momentum,axis=0)
        #     exttau=np.sum(self.depend[i,self.Ndimk:]*imagtime.T)
        #     extind=sublatind[i]*2+sublatind[self.perm[i]]
        #     res[i]=res[i]*self.ut[l,exttau]*self.kbasis[i,extk[0],extk[1],extk[2]]*self.sublatbasis[j,extind]
        

        # getting the indices for the cut G. However we have to get these for the sigma
        # since the sigma has the same indices as the basis functions.
        k = np.sum(self.depend[:, :self.Ndimk, None] * momentum[None,:,:], axis=1)# this is the k we need. should put it in 1BZ.
        # note for sigsym: if 11 and 22, mod=0, 1-0*2=1; if 12 or 21, mod=1,1-1*2=-1. ofcourse it is an array with nGf elements.
        sigsym=1-((sublatind-sublatind[self.perm])%2)*2
        kfactor=(sigsym)**((np.mod(k[:,0], self.knum)-(k[:,0]))/self.knum+(np.mod(k[:,1], self.knum)-(k[:,1]))/self.knum+(np.mod(k[:,2], self.knum)-(k[:,2]))/self.knum)
        sigk=np.mod(k,self.knum)
        tau = -1*np.sum(self.depend[:,self.Ndimk:] * imagtime.T, axis=1)
        oppo_taulimit=1-self.taulimit# have to take the correct limit.
        tfactor=(-1)**((np.mod(tau-oppo_taulimit, self.taunum)-(tau-oppo_taulimit))/(self.taunum))
        sigtau=np.mod(tau-oppo_taulimit, self.taunum)+oppo_taulimit
        
        
        # sigind = (2-sublatind) * 2 + (2-sublatind[self.perm])
        sigind=(sublatind[self.perm]-1)*2+sublatind-1# here 1 and 2 are used for up and dn. 2-1=1.2-2=0.
        zero_indices = np.where(tau == 0)[0]
        res[zero_indices]*=self.taulimitfactor[zero_indices]
        # if np.any(tau==0) or np.any(tau==self.taunum):#
        # print('\nchecking indices:')
        # print('momentum=\n',momentum)
        # print('k=\n',k)

            # print('\nimagtime=',imagtime.T)
            # print('tau=',tau)
            # print('sigtau=',sigtau)

            # print('sublatind=',sublatind)
        # print('ind for sigma:',sigind)
        
        # print('sigsym=',sigsym)
        # print('kfactor=',kfactor)
            # print('oppo_limit',oppo_taulimit)
            # print('tfactor=',tfactor)
            # print('slicdG:',slices)
        # print('result w/o basis:',res)
        # print('check',np.prod(slices)/slices[0],np.prod(slices)/slices[1],np.prod(slices)/slices[2],np.prod(slices)/slices[3],np.prod(slices)/slices[4],np.prod(slices)/slices[5])
        # Update res with vectorized operations



        res *= (self.ut[l, sigtau] * 
            self.kbasis[i_coeff, sigk[:, 0], sigk[:, 1], sigk[:, 2]] * 
            self.sublatbasis[j_coeff, sigind]*tfactor*kfactor)#
        # return res[1]

        return np.sum(res)/self.symfactor

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
    
