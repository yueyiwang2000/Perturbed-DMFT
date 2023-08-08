import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess

def run(list):
    for (U,T) in list:
        print("U=",U,'T=',T)
        cmd='python launch_pert_ctqmc.py {} {} 0'.format(U,T)
        subprocess.call(cmd, shell=True)
        cmd='python launch_pert_ctqmc.py {} {} 1'.format(U,T)
        subprocess.call(cmd, shell=True)

list_3=((3,0.13),(3,0.14),(3,0.12))
list_4=((4.,0.21),(4.,0.19),(4.,0.2))
list_5=((5.,0.28),(5.,0.29),(5.,0.27))#
list_6=((6.,0.34),(6.,0.35),(6.,0.36),(6.,0.37),(6.,0.38))
list_7=((7.0,0.37),(7.0,0.38),(7.0,0.39),(7.0,0.4))
# list_7=((7.0,0.43),(7.0,0.39),(7.0,0.41),(7.0,0.4),(7.0,0.42))
list_8=((8.0,0.43),(8.0,0.44),(8.0,0.45),(8.0,0.46))
# run(list_2)
run(list_7)
run(list_3)

# run(list_8)
# run(list_6)
# run(list_4)
# run(list_5)
# subprocess.call(cmd, shell=True)
# > ./files/{}_{}/{}_{}.txt