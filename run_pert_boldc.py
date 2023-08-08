import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess

def run(list):
    for (U,T) in list:
        print("U=",U,'T=',T)
        cmd='python launch_pert_boldc.py {} {} 0'.format(U,T)
        subprocess.call(cmd, shell=True)
        cmd='python launch_pert_boldc.py {} {} 1'.format(U,T)
        subprocess.call(cmd, shell=True)


# list_7=((7.0,0.46),(7.0,0.45))
list_10=((10.0,0.48),(10.0,0.49),(10.0,0.5))

# run(list_7)
run(list_10)
# subprocess.call('python run_pert_ctqmc.py', shell=True)