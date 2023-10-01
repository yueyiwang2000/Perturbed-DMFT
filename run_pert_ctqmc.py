import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess

def run(list):
    for (U,T) in list:
        print("U=",U,'T=',T)
        cmd='python launch_pert_ctqmc.py {} {} 1'.format(U,T)
        subprocess.call(cmd, shell=True)
        cmd='python launch_pert_ctqmc.py {} {} 0'.format(U,T)
        subprocess.call(cmd, shell=True)
list_test=((3.,0.13),(5.,0.29))
list_2=((2.,0.08),(2.,0.1))
list_3=((3.,0.07),(3.,0.09),(3.,0.11))
list_4=((4.,0.17),(4.,0.19),(4.,0.21),(2.,0.1))
list_5=((5.,0.26),(5.,0.24),(5.,0.27))#
list_6=((6.,0.31),(6.,0.33),(6.,0.35),(6.,0.37))
list_7=((7.0,0.36),(7.0,0.38),(7.0,0.4),(7.0,0.42))
list_8=((8.0,0.4),(8.0,0.42),(8.0,0.45),(8.0,0.46))
# run(list_test)
# run(list_2)
# run(list_3)
# run(list_4)
# run(list_5)
# run(list_6)
run(list_7)
run(list_8)


# subprocess.call(cmd, shell=True)
# > ./files/{}_{}/{}_{}.txt