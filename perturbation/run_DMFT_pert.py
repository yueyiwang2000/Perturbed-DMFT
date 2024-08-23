import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess

'''
this code is written to generate all observables for perturbation.
'''

def run_pert(U,T):
    cmd='python pert_DMFT_PM.py {} {}'.format(U,T)
    subprocess.call(cmd, shell=True)

def runmag():
    T_bound=np.array(((3.0,0.07,0.14),(4.,0.15,0.25),(5.,0.2,0.31),(6.,0.27,0.37),(7.,0.27,0.3),
                      (8.,0.3,0.45),(9.,0.28,0.32),(10.,0.3,0.5),(11.,0.3,0.5),(12.,0.3,0.5),(13.,0.3,0.5),(14.,0.25,0.45)))
    for list in T_bound:
        U=list[0]
        # print(U)
        if U>=6 and U<8:
            for T in np.arange(int(list[1]*100),int(list[2]*100))/100:
                print(U,T)
                run_pert(U,T)
                dir='./magdata/{}_{}.dat'.format(U,T)
                data=np.loadtxt(dir)
                if data[1,2]>data[1,3] and data[1,3]>data[1,4]:# order up, mag dn ==>PM
                    break
def runthermo():
    T_bound=np.array(((8.,0.25,0.45),(9.,0.28,0.32),(10.,0.31,0.5),(11.,0.3,0.5),(12.,0.3,0.6),(13.,0.3,0.5),(14.,0.25,0.5)))
    for list in T_bound:
        U=list[0]
        # print(U)
        if U==12.0 or U==14.0:
            for T in np.arange(int(list[1]*100),int(list[2]*100))/100:
                print(U,T)
                run_pert(U,T)
                dir='./energydata/{}_{}.dat'.format(U,T)
                # data=np.loadtxt(dir)
                # if data[1,2]>data[1,3] and data[1,3]>data[1,4]:# order up, mag dn ==>PM
                #     break

if __name__ == "__main__":
    runthermo()
    # run_pert(10.0,0.1)
    # run_pert(8.0,0.15)
    # run_pert(8.0,0.25)
    # run_pert(8.0,0.05)