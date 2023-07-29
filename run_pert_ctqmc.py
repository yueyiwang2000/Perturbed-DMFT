import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess

def run(list):
    for (U,T) in list:
        print("U=",U,'T=',T)
        cmd='python launch_pert_ctqmc.py {} {} '.format(U,T)
        subprocess.call(cmd, shell=True)

# list_1=((1.,0.05),(1.,0.03),(1.,0.02),(1.,0.015),(1.,0.01))
# list_half=((0.5,0.01),(0.5,0.005),(0.5,0.008),(0.5,0.015),(0.5,0.02))
# list_2=((2.,0.04),(2.,0.05))
list_3=((3,0.12),(3,0.08),(3,0.1))
list_4=((4.,0.21),(4.,0.19),(4.,0.2))
list_5=((5.,0.28),(5.,0.29))#,(5.,0.27)
list_6=((6.,0.35),(6.,0.36))#(6.,0.37),(6.,0.38),
list_7=((7.0,0.3),(7.0,0.4),(7.0,0.35))
# list_7=((7.0,0.43),(7.0,0.39),(7.0,0.41),(7.0,0.4),(7.0,0.42))
# run(list_2)
# run(list_3)
# run(list_4)
# run(list_5)
# run(list_6)
run(list_3)
run(list_7)
# cmd='python iterate_backup.py > ./files/bethe_3/bethe_3.txt'
# cmd='python iterate_cubic.py > ./files/cubic/cubic.txt'
# subprocess.call(cmd, shell=True)
# > ./files/{}_{}/{}_{}.txt