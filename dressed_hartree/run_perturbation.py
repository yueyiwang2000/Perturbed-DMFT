import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess

def run(B,U,T,order=3):
    print("U=",U,'T=',T,'B=',B)
    cmd='python pert_dressed_hartree_var.py {} {} {} {}'.format(B,U,T,order)
    subprocess.call(cmd, shell=True)

def runcountB0(B,U,T,order=3):
    print("U=",U,'T=',T,'B=',B,'order=',order)
    cmd='python pert_dressed_hartree_var.py {} {} {} {} {}'.format(B,U,T,order,0)
    subprocess.call(cmd, shell=True)

def runB0(B,U,T,order=3):
    print("U=",U,'T=',T,'B=',B)
    cmd='python pert_dressed_hartree_varB0.py {} {} {} {}'.format(B,U,T,order)
    subprocess.call(cmd, shell=True)

def runsus(B,U,T,order=3):
    print("U=",U,'T=',T,'B=',B)
    cmd='python pert_dressed_hartree_varB0.py {} {} {} {}'.format(0.0,U,T,order)
    subprocess.call(cmd, shell=True)

T1=np.array([0.24,0.26,0.28,0.3,0.32,0.34,0.36,0.38,0.4,0.42,0.44])
T2=np.array([0.24,0.26,0.28,0.3,0.32,0.34,0.36,0.38,0.4])
T3=np.array([0.24,0.26,0.28,0.3,0.32,0.34,0.36])
T4=np.array([0.24,0.26,0.28,0.3,0.32,0.34,0.36])
T5=np.array([0.2,0.25,0.3,0.35,0.4])
T6=np.array([0.42,0.44,0.46,0.48,0.5])
Tlist2=(np.arange(30)+30)/100
# B=0.01
Blist=(np.arange(2))/500
U=7.0
Blist_35=(np.arange(8)+27)/1000
# Blist_40=(np.arange(10)+10)/2000
for T in Tlist2:
    for B in Blist:
        runcountB0(B,U,T,3)
        runcountB0(B,U,T,2)
        runcountB0(B,U,T,1)
        runcountB0(B,U,T,0)

