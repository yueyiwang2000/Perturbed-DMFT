# here I just copied my function for Hilbert transformation
# which is used for green's function to here.
import numpy as np
from scipy import integrate
def find_index(myelist,val):# to find the index of lower/upper bound....
    for i in np.arange(np.size(myelist)):
        if myelist[i]>val:
            if myelist[i]-val>val-myelist[i-1]:
                return i-1
            else:
                return i
    return np.size(myelist)-1

def Hilbert_Transformation(z,myelist,mydos,threshould=0.1):
    Integration_sum=0+0j
    length=len(myelist)
    # egrid=(myelist[-1]-myelist[0])/length# interval of e. usually (6-(-6))/len.
    # lowerbound=np.maximum(myelist[0],z.real-threshould)
    # upperbound=np.minimum(myelist[-1],z.real+threshould)
    # flag=0# to see if we need this 'add back' to improve the accuracy
    # firsttime=0
    # for i in np.arange(length//2-1)*2:
    # lower_index=find_index(myelist,lowerbound)
    # upper_index=find_index(myelist,upperbound)
    Integration_sum=integrate.simps(mydos*myelist/(myelist**2-z**2),myelist)# this is correct
    # for i in np.arange(length-1):
    #     if i<lower_index or i>upper_index:
    #         Integration_sum=Integration_sum+egrid*(mydos[i]/(z-myelist[i]) + mydos[i+1]/(z-myelist[i+1]))/2 # Newton's method.
    #     else:
    #         flag=1
    #         if firsttime==0:
    #             zero_index=find_index(myelist,z.real)
    #             dos0=mydos[zero_index]
    #         firsttime=1
    #         # print(z-myelist[i])
    #         Integration_sum=Integration_sum+egrid*((mydos[i]-dos0)/(z-myelist[i]) + (mydos[i+1]-dos0)/(z-myelist[i+1]))/2 # Newton's method.
    # if flag==1:
    #     addback=np.log((z-lowerbound)/(z-upperbound))
    #     # print(addback)
    #     Integration_sum+=dos0*addback
    return Integration_sum