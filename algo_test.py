import numpy as np
import time

def dispersion(kx,ky,kz,a=1,t=1):
    e_k=-2*t*np.cos(kx*a)-2*t*np.cos(ky*a)-2*t*np.cos(kz*a)
    return e_k


def alpha(beta,mu,sig):
    # here iomega should be ij*omega.
    n=np.size(sig)# same numbers of matsubara freqs as sigma file
    iom = 1j*(2*np.arange(n)+1)*np.pi/beta# Matsubara freqs
    return iom+mu*np.ones_like(sig)-sig


start_time = time.time()



#1
klist=(np.arange(19)-9)/10*np.pi
sigma=4.0*np.ones(2000)
sum=0
for kx in klist:
    for ky in klist:
        for kz in klist:
            G_k=1/(alpha(100,0,sigma)-dispersion(kx,ky,kz))
            G_kq=1/(alpha(100,0,sigma)-dispersion(kx,ky,kz))
            sum+=G_k*G_kq

#2



end_time = time.time()


elapsed_time = end_time - start_time
print("time is {:.6f} s".format(elapsed_time))
