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

list_6=((6.0,0.4),(6.0,0.42),(6.0,0.44))
list_7=((7.0,0.48),(7.0,0.5))#,(7.0,0.43),(7.0,0.41)
list_10=((10.0,0.46),(10.0,0.48),(10.0,0.5))#
list_11=((11.0,0.45),(11.0,0.46),(11.0,0.47),(11.0,0.48))
list_12=((12.0,0.43),(12.0,0.44),(12.0,0.45),(12.0,0.46),(12.0,0.47))
list_13=((13.0,0.42),(13.0,0.43),(13.0,0.44),(13.0,0.45))
# list_14=((14.0,0.41),(14.0,0.42),(14.0,0.43),(14.0,0.44),(14.0,0.45))
# list_15=((15.0,0.39),(15.0,0.38),(15.0,0.42),(15.0,0.43),(15.0,0.44))#
# run(list_6)
run(list_7)
run(list_10)
# run(list_12)
run(list_13)
# run(list_14)

# subprocess.call('python run_pert_ctqmc.py', shell=True)