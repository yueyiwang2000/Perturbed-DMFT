import numpy as np
import matplotlib.pyplot as plt
import sys


def plot_sigma(filename):
    sigma=np.loadtxt(filename)
    omega=sigma[:,0]
    for i in np.arange(np.size(sigma[1,:])-1)+1:
        plt.plot(omega,sigma[:,i],label='{}th column in {}'.format(i,filename))
    plt.legend()


filenum=len(sys.argv)-1
for i in np.arange(filenum):
    plot_sigma(sys.argv[i+1])
plt.show()