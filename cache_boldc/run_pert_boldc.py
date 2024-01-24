import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess

def run1(list):
    for (U,T) in list:
        # print("U=",U,'T=',T)
        # cmd='python launch_pert_boldc.py 3 {} {} 1'.format(U,T)
        # subprocess.call(cmd, shell=True)
        # cmd='python launch_pert_boldc.py 2 {} {} 1'.format(U,T)
        # subprocess.call(cmd, shell=True)
        cmd='python launch_pert_boldc.py 2 {} {} 0'.format(U,T)
        subprocess.call(cmd, shell=True)

def run2(list):
    for (U,T) in list:
        print("U=",U,'T=',T)
        # cmd='python launch_pert_boldc.py 2 {} {} 0'.format(U,T)
        # subprocess.call(cmd, shell=True)        
        cmd='python launch_pert_boldc.py 3 {} {} 1'.format(U,T)
        subprocess.call(cmd, shell=True)
        # cmd='python launch_pert_boldc.py 2 {} {} 0'.format(U,T)
        # subprocess.call(cmd, shell=True)
        # cmd='python launch_pert_boldc.py 2 {} {} 1'.format(U,T)
        # subprocess.call(cmd, shell=True)
list_critical=((10.0,0.4733),(10.0,0.473))
list_afm=((4.0,0.15),(8.0,0.35),(13.0,0.4))
list_3=((3.0,0.1),(3.0,0.11),(3.0,0.12),(3.0,0.13))
list_4=((4.0,0.18),(4.0,0.19),(4.0,0.20),(4.0,0.21),(4.0,0.22))
list_5=((5.0,0.22),(5.0,0.24),(5.0,0.26),(5.0,0.28))
list_6=((6.0,0.33),(6.0,0.34),(6.0,0.35))
# list_7=((7.0,0.39),(7.0,0.42))#(7.0,0.36),(7.0,0.37),
list_7_5=((7.5,0.46),(7.5,0.47),(7.5,0.48),(7.5,0.49))#
list_8=((8.0,0.42),(8.0,0.40),(8.0,0.41),(8.0,0.43),(8.0,0.44))
list_9=((9.0,0.45),(9.0,0.46),(9.0,0.47),(9.0,0.48))
# list_10=((10.0,0.473),(10.0,0.474))#
list_10=((10.0,0.43),(10.0,0.44),(10.0,0.45),(10.0,0.46),(10.0,0.47),(10.0,0.48),(10.0,0.49),(10.0,0.5))
list_11=((11.0,0.45),(11.0,0.46),(11.0,0.47),(11.0,0.48))
list_12=((12.0,0.43),(12.0,0.44),(12.0,0.45),(12.0,0.46),(12.0,0.47))
list_13=((13.0,0.42),(13.0,0.43),(13.0,0.44),(13.0,0.45))
list_14=((14.0,0.37),(14.0,0.38),(14.0,0.39),(14.0,0.4))#,(14.0,0.44),(14.0,0.45)
list_15=((15.0,0.35),(15.0,0.36),(15.0,0.37),(15.0,0.38))#(15.0,0.39),
#

#everytime after run, comment the command below!
# run2(list_critical)
run1(list_10)
# run1(list_7_5)
# run1(list_6)
# run2(list_5)
# subprocess.call('python run_pert_ctqmc.py', shell=True)
# run1(list_14)
# run1(list_15)