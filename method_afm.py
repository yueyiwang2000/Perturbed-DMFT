import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import cmath
from numba import jit

@jit(nopython=True)

def oned_dispersion(k,a,t):
    e_k=-2*t*np.cos(k*a)
    return e_k

def put_in(emin,egrid,x):# figure out which grid should x be put
    if x < emin:
        print('error')
        return -1
    return int((x-emin)/egrid)



def cubic_dos(e_grid,a=1,t=1):# in default, t=1. a is lattice constant. In default is 1.
    emin=-6*t
    emax=6*t
    dos=np.zeros(int((emax-emin)/e_grid)+1)
    knum=500
    k_grid=2*np.pi/a/knum
    kmin=-np.pi/a
    kmax=np.pi/a
    elist=np.linspace(emin,emax,int((emax-emin)/e_grid)+1)
    klist=np.linspace(kmin,kmax,int((kmax-kmin)/k_grid)+1)
    # this is the conceptually easiest way to calculate.
    for kx in klist:
        ex=oned_dispersion(kx,a,t)
        for ky in klist:
            ey=oned_dispersion(ky,a,t)
            for kz in klist:
                ez=oned_dispersion(kz,a,t)
                dos[put_in(emin,e_grid,ex+ey+ez)]+=1#/knum/knum/knum/e_grid
    dos=(dos+dos[::-1])/2#symmetrize
    dos=dos/integrate.simpson(dos,elist)#normalize
    for i in np.arange(np.size(elist)):
        print(elist[i],dos[i])
    # plt.plot(elist,dos)
    # plt.show()
    return elist,dos

def bethe_dos(egrid):
    emin=-1
    emax=1
    dos=np.zeros(int((emax-emin)//egrid)+1)
    elist=np.linspace(emin,emax,int((emax-emin)//egrid)+1)
    for i in np.arange(np.size(dos)):
        dos[i]=2/np.pi*np.sqrt(1-elist[i]**2)
        print(elist[i],dos[i])
    # plt.plot(elist,dos)
    # plt.show()
    return elist,dos

def find_index(myelist,val):# to find the index of lower/upper bound....
    for i in np.arange(np.size(myelist)):
        if myelist[i]>=val:
            if myelist[i]-val>val-myelist[i-1]:
                return i-1
            else:
                return i
    return np.size(myelist)-1


def Hilbert_Transformation(z,myelist,mydos,threshould=0):
    Integration_sum=0+0j
    length=len(myelist)
    egrid=(myelist[-1]-myelist[0])/length# interval of e. usually (6-(-6))/len.
    # lowerbound=np.maximum(myelist[0],z.real-threshould)
    # upperbound=np.minimum(myelist[-1],z.real+threshould)
    mydos=mydos+mydos[::-1]
    mydos=mydos/integrate.simps(mydos,myelist)
    # for i in np.arange(length//2-1)*2:
    # lower_index=find_index(myelist,lowerbound)
    # upper_index=find_index(myelist,upperbound)
    # print(lower_index,upper_index)
    Integration_sum=integrate.simpson(mydos/(z-myelist),myelist)# this is correct
    # if lowerbound>myelist[0]:
    # Integration_sum+=integrate.simps(mydos[:lower_index]/(z-myelist[:lower_index]),myelist[:lower_index])
    # if upperbound<myelist[-1]:
    # Integration_sum+=integrate.simps(mydos[upper_index:]/(z-myelist[upper_index:]),myelist[upper_index:])
    # if lowerbound<upperbound:
    #     # print('lowerbound<upperbound')
    #     zero_index=find_index(myelist,z.real)
    #     dos0=mydos[zero_index]
    #     Integration_sum+=integrate.simps((mydos[lower_index:upper_index+1]-dos0)/(z-myelist[lower_index:upper_index+1]),myelist[lower_index:upper_index+1])
    #     Integration_sum+=dos0*cmath.log((z-myelist[lower_index])/(z-myelist[upper_index]))



    # flag=0# to see if we need this 'add back' to improve the accuracy
    # firsttime=0
    # for i in np.arange(length-1):
    #     if i<=lower_index or i>=upper_index:
    #         Integration_sum=Integration_sum+egrid*(mydos[i]/(z-myelist[i]) + mydos[i+1]/(z-myelist[i+1]))/2 # Newton's method.
    #     else:
    #         flag=1
    #         if firsttime==0:
    #             zero_index=find_index(myelist,z.real)
    #             dos0=mydos[zero_index]
    #         firsttime=1
    #         Integration_sum=Integration_sum+egrid*((mydos[i]-dos0)/(z-myelist[i]) + (mydos[i+1]-dos0)/(z-myelist[i+1]))/2 # Newton's method.
    # if flag==1:
    #     addback=np.log((z-lowerbound)/(z-upperbound))
    #     Integration_sum+=dos0*addback
    return Integration_sum


# def W():





def test_Hil_trans(size,beta,mu=0):
    # to generate GF
    GreenFunction=np.zeros(size,dtype=complex)
    # dosdata=np.loadtxt('cubic_dos.txt').T
    dosdata=np.loadtxt('DOS.dat').T
    elist=dosdata[0]
    dos=dosdata[1]
    # print('total=',integrate.simps(dos,elist))
    # dos=dos/integrate.simps(dos,elist)
    # print('total=',integrate.simps(dos,elist))
    # myelist,bethedos=cubic_dos(0.002)
    omegalist=(2*np.arange(size)+1)*np.pi/beta# matsubara freq. take beta=100
    simple_sigma=0.5
    zlist=(1j*omegalist)/2#+mu-simple_sigma# actually this should be a z list. which is i*omega+mu-sigma
    deltalist=np.zeros(size,dtype=complex)
    for n in np.arange(size):
        GreenFunction[n]=Hilbert_Transformation(zlist[n],elist,dos)
    deltalist=zlist-1.0/GreenFunction
    # plt.plot(omegalist.real,iomlist.real,label='iom real')
    # plt.plot(omegalist.real,iomlist.imag,label='iom imag')
    # plt.plot(omegalist.real,(1/GreenFunction).real,label='1/G real')
    # plt.plot(omegalist.real,(1/GreenFunction).imag,label='1/G imag')
    plt.plot(omegalist.real,deltalist.real,label='delta real')
    plt.plot(omegalist.real,deltalist.imag,label='delta imag')
    plt.legend()
    plt.show()
    # for i in np.arange(size):
    #     print(omegalist[i],deltalist[i].real,deltalist[i].imag)
    return 0


def gen_trial_sigma_para(b):
    for i in np.arange(2000):
        omega=(2*i+1)*np.pi/b
        print(omega, 0.5, 0, -0.5, 0)
    return 0


def complex_sqrt(z1,z2):
    angles=np.angle([z1,z2])
    zangle=(angles[0]+angles[1])/2
    znorm=np.sqrt(np.abs(z1)*np.abs(z2))
    return znorm*np.cos(zangle)+1j*znorm*np.sin(zangle)



if __name__ == "__main__":
    # myelist,bethedos=bethe_dos(0.001)
    # plt.plot(myelist,bethedos)
    # plt.show()
    # test_Hil_trans(1000,100)
    cubic_dos(0.01)
    # gen_trial_sigma_para(100)
    # print(complex_sqrt(-1+100j,1+100j))