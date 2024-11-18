import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
'''
This code is a simple exercise to check how to unfold band structure of a t-t' tight-biding model on a 2D square lattice.
'''



knum=1000
kxindlist=(np.arange(knum)-(knum-1)/2)
kyindlist=(np.arange(knum)-(knum-1)/2)
kxlist=(np.arange(knum)-(knum-1)/2)/knum*2*np.pi
kylist=(np.arange(knum)-(knum-1)/2)/knum*2*np.pi

kx, ky= np.meshgrid(kxlist, kylist, indexing='ij')
t1=1
t2=1/2
filling=0.5
eps1=-2*t1*(np.cos(kx)+np.cos(ky))
eps2=-2*t2*(np.cos(kx+ky)+np.cos(kx-ky))
Hplus=eps1+eps2
Hminus=-eps1+eps2
Emin=min(np.min(Hplus),np.min(Hminus))
Emax=max(np.max(Hplus),np.max(Hminus))
print('Emin=',Emin,'Emax=',Emax)
#search for fermi surface of H+, which is in single unit cell.
Eupper=Emax
Elower=Emin
for i in np.arange(20):
    Etrial=(Eupper+Elower)/2
    # print('Etrial=',Etrial)
    ptnums=np.sum(Hplus < Etrial)
    if ptnums>filling*knum**2:# too much. the fermi energy should be lower.
        Eupper=Etrial
    else:
        Elower=Etrial

Eupper=Emax
Elower=Emin
for i in np.arange(20):
    Etrialp=(Eupper+Elower)/2
    # print('Etrialp=',Etrialp)
    ptnums=np.sum((Hplus < Etrialp) & (np.abs(kx)+np.abs(ky)<np.pi))       +         np.sum((Hminus < Etrialp) & (np.abs(kx)+np.abs(ky)<np.pi))
    if ptnums>filling*knum**2:# too much. the fermi energy should be lower.
        Eupper=Etrialp
    else:
        Elower=Etrialp


threshold=0.03
cmap = ListedColormap(['white', 'white', 'red'])
cmap2 = ListedColormap(['white', 'white', 'blue'])
cmap3 = ListedColormap(['white', 'white', 'black'])
mask = np.abs(Hplus - Etrial) < threshold
mask2 = ((np.abs(Hplus - Etrialp) < threshold)|(np.abs(Hminus - Etrialp) < threshold)) &(np.abs(kx)+np.abs(ky)<np.pi)
mask3 = np.abs(np.abs(kx)+np.abs(ky)-np.pi)<0.02
# mask2 = np.abs(Hplus - Etrial) < threshold
# plt.imshow(Hplus, cmap='hot', interpolation='nearest')
plt.imshow(mask3, cmap=cmap3, interpolation='nearest', alpha=1)
plt.imshow(mask, cmap=cmap, interpolation='nearest', alpha=0.7)
plt.imshow(mask2, cmap=cmap2, interpolation='nearest', alpha=0.5)

plt.show()
