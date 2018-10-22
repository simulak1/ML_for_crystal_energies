'''
This module is for the construction of Ewald matrices.
The matrices are computed based on the atom coordinates 
and atomic numbers (nuclear charges) of the crystal data.
The calculation is performed as in the article of this link:

https://onlinelibrary.wiley.com/doi/full/10.1002/qua.24917

The actual computations are done in a c library for speeding
up computations.
'''

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import scipy as sp
import scipy.special
import sys
import time
sys.path[0]="/u/82/simulak1/unix/Desktop/doctoral_studies/kurssit/ML_in_MS/c_functions"
import _C_arraytest
sys.path[0]="/u/82/simulak1/unix/Desktop/doctoral_studies/kurssit/ML_in_MS/src"
import data

def SortRows(M):
    indexlist = np.argsort(np.linalg.norm(M,axis=1))
    M=M[indexlist,:]
    return M[:,indexlist]

def compute_ewald_matrices(N,Lcut,Gcut):
    DataPointIndices=np.arange(0,N)#np.random.randint(0,2400,size=N)
    materials=data.load_data(DataPointIndices)
    
    M=np.zeros((80,80))

    for i in range(N):
        print("Computing structure "+str(i+1)+"/"+str(N)+" ; Saving matrix "+str(DataPointIndices[i]))
        a=np.sqrt(np.pi)*(0.01*materials[i].N/np.dot(materials[i].La[0,:],np.cross(materials[i].La[1,:],materials[i].La[2,:])))**(1/6)
        
        M=_C_arraytest.make_ewald_matrix(materials[i].N,materials[i].La,materials[i].xyz,materials[i].Z,Gcut,Lcut,a)
        M=SortRows(M)
        
        np.save("../data/ewald_matrices/"+str(DataPointIndices[i]),M)

t=time.time()
compute_ewald_matrices(2400,7,5)
print("Time taken: "+str(time.time()-t))

