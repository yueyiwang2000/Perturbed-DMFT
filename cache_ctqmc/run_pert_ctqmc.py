import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess

def run(list):
    for (U,T) in list:
        print("U=",U,'T=',T)

        cmd='python launch_pert_ctqmc.py {} {}'.format(U,T)
        subprocess.call(cmd, shell=True)
list_test=((3,0.1),(5.,0.29),(6.,0.37))#run 100+steps!
# list_2=((2.,0.08),(2.,0.1))
list_3=((3.,0.08),(3.,0.09),(3.,0.1),(3.,0.11),(3.,0.12),(3.,0.13))
list_4=((4.,0.1),(5.,0.15))
# list_5=((5.,0.27),(5.,0.3),(5.,0.29))#
list_6=((6.,0.15),(6.,0.2))#
# list_7=((7.0,0.38),(7.0,0.39),(7.0,0.4),(7.0,0.41))
# list_8=((8.0,0.4),(8.0,0.42),(8.0,0.45),(8.0,0.46))
# list_9=((9.0,0.48),(9.0,0.42),(9.0,0.45),(9.0,0.46))
# list_10=((10.0,0.44),(10.0,0.46),(10.0,0.48),(10.0,0.5))
# list_12=((12.0,0.42),(12.0,0.44))
# list=((10.0,0.3),(10.0,0.2),(10.0,0.1),(8.0,0.5),(8.0,0.4),(8.0,0.3),(8.0,0.2),(8.0,0.1))
# run(list_test)
# run(list_2)
# run(list_3)
# run(list_4)
# run(list_5)
# run(list_6)
# run(list_7)
# run(list_8)
# run(list_9)
# run(list_3)
run(list_6)

# subprocess.call(cmd, shell=True)
# > ./files/{}_{}/{}_{}.txt