import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess

def run(B,U,T,order=3):

    # cmd='python pert_dressed_hartree_var.py {} {} {} {}'.format(B,U,T,order)
    # cmd='python pert_strictorder.py {} {} {} {}'.format(B,U,T,order)
    for ifit in np.arange(2):
        for countB in np.arange(2):
            cmd='python pert_strategies.py {} {} {} {} {} {}'.format(B,U,T,order,countB,ifit)
            subprocess.call(cmd, shell=True)
    # cmd='python pert_strategies.py {} {} {} {} {} {}'.format(B,U,T,order,1,0)
    # subprocess.call(cmd, shell=True)

def run_alpha(U,T,alpha,order):
    for ifit in np.arange(2):
        cmd='python pert_strategies_alpha.py {} {} {} {} {}'.format(U,T,alpha,order,ifit)
        subprocess.call(cmd, shell=True)


U=7.0
Tlist_7=       np.array([0.2,0.25,0.3,0.32,0.35,0.37,0.39])
pointsnumlist7=np.array([80 ,60  ,40 ,40  ,30  ,15  ,16])
b_resume7=np.array([51,40,20,31])/500
b_stop7=       np.array([0.158,0.11,0.066,0.06,0.048,0.04,0.03])

Tlistalpha_7=       np.array([0.2,0.3,0.32,0.35,0.37,0.39])
alpha_list=np.arange(11)/10
alphalist2=(np.arange(20)+80)/100

# for it,T in enumerate(Tlist_7):
#     if T==0.37 or T==0.32:
#         for B in np.arange(pointsnumlist7[it])/500:
#             if B<=b_stop7[it]:
#                 run(B,U,T,0)
#                 run(B,U,T,1)
#                 run(B,U,T,2)
#                 run(B,U,T,3)
alpha_list[0]=0.0
# print(alpha_list)
for it,T in enumerate(Tlistalpha_7):
    if T==0.37:
        for alpha in alphalist2:
            # run_alpha(U,T,alpha,0)
            # run_alpha(U,T,alpha,1)
            # run_alpha(U,T,alpha,2)
            run_alpha(U,T,alpha,3)