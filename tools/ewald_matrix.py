'''
This module is for the construction of Ewald matrices.
The matrices are computed based on the atom coordinates 
and atomic numbers (nuclear charges) of the crystal data.
The calculation is performed as in the article of this link:

https://onlinelibrary.wiley.com/doi/full/10.1002/qua.24917

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
sys.path="$PWD"


def short_range(xi,xj,Zi,Zj,L,Lmax,a):
    '''
    The short-range component of the ewald sum. The summation
    is performed between two atoms and their periodic images.
    The periodic images are accounted from as far as a chosen 
    cutoff value 'Lmax' allows.
    '''
    # The short range sum
    xij=0

    # The lattice vectors
    Lx=L[0,:]
    Ly=L[1,:]
    Lz=L[2,:]

    # The maximum distance at each of the lattice vector directions.
    # This method is a guess to include all the lattice vectors, meaning
    # that the number of images considered is made ridicously high. Think of
    # a more elegant method.
    if(LA.norm(Lx)>Lmax or LA.norm(Ly)>Lmax or LA.norm(Lz)>Lmax):sys.exit("ERROR. SHORT_RANGE: The cutoff is too small.")
    Nx=int((Lmax-Lmax%LA.norm(Lx))/LA.norm(Lx))+1
    Ny=int((Lmax-Lmax%LA.norm(Ly))/LA.norm(Ly))+1
    Nz=int((Lmax-Lmax%LA.norm(Lz))/LA.norm(Lz))+1
            
    
    # The summing over periodic images
    for i in range(-Nx,Nx):
        for j in range(-Ny,Ny):
            for k in range(-Nz,Nz):
                Limage=i*Lx+j*Ly+k*Lz
                dist=LA.norm(xi-xj+Limage)
                if(dist<Lmax): # If inside cutoff
                    xij=xij+scipy.special.erfc(a*dist)/dist
                    
    xij=Zi*Zj*xij
    return xij

def long_range(xi,xj,Zi,Zj,L,Gmax,a):
    '''
    The long-range component of the Ewald sum. The summation goes 
    over images of the reciprocal simulation cell. Care must be taken
    in choosing the cutoff. Could DFT cutoffs give us a hint?
    '''
    # The long range sum
    xij=0

    # The lattice vectors
    Lx=L[0,:]
    Ly=L[1,:]
    Lz=L[2,:]

    # The reciprocal lattice vectors
    Gx=np.pi*np.cross(Ly,Lz)/np.dot(Lx,np.cross(Ly,Lz))
    Gy=np.pi*np.cross(Lz,Lx)/np.dot(Ly,np.cross(Lz,Lx))
    Gz=np.pi*np.cross(Lx,Ly)/np.dot(Lz,np.cross(Lx,Ly))

    # The maximum distance at each of the lattice vector directions.
    # This method is a guess to include all the lattice vectors, meaning
    # that the number of images considered is made ridicously high. Think of
    # a more elegant method.
    if(LA.norm(Gx)>Gmax or LA.norm(Gy)>Gmax or LA.norm(Gz)>Gmax):sys.exit("ERROR. LONG_RANGE: The cutoff is too small.")
    Nx=int((Gmax-Gmax%LA.norm(Gx))/LA.norm(Gx))+10
    Ny=int((Gmax-Gmax%LA.norm(Gy))/LA.norm(Gy))+10
    Nz=int((Gmax-Gmax%LA.norm(Gz))/LA.norm(Gz))+10

    # The summing over periodic images
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                Gimage=i*Gx+j*Gy+k*Gz
                if(LA.norm(Gimage)<Gmax and i>0 and j>0 and k>0): # If inside cutoff
                    xij=xij+np.exp(-(LA.norm(Gimage)**2)/(2*a)**2)/(LA.norm(Gimage)**2)*np.cos(np.dot(Gimage,xi-xj))
                    
    xij=Zi*Zj*xij/(np.pi*np.dot(Lx,np.cross(Ly,Lz)))
    return xij

def self_correction(xi,xj,Zi,Zj,L,a):
    '''
    This is somehow meant to make the system neutral in terms of 
    a compensating background charge.
    '''
    # The lattice vectors
    Lx=L[0,:]
    Ly=L[1,:]
    Lz=L[2,:]
    # The volume
    V=np.dot(Lx,np.cross(Ly,Lz))
    return -(Zi**2+Zj**2)*a/np.sqrt(np.pi)-(Zi+Zj)**2*np.pi/(2*V*a)

def diagonal(Z,L,a):
    '''
    The diagonal part of the Ewald matrix.
    '''   
    # The lattice vectors
    Lx=L[0,:]
    Ly=L[1,:]
    Lz=L[2,:]
    # The volume
    V=np.dot(Lx,np.cross(Ly,Lz))

    return -Z**2*a/np.sqrt(np.pi)-Z**2*np.pi/(2*V*a**2)

def make_ewald_matrix(N,L,xyz,Z,Lmax,Gmax,a):
    '''
    Constructs a single Ewald matrix. 
    Input:
    * N,    number of atoms
    * L,    the matrix of lattice vectors
    * xyz,  the coordinates of the atoms
    * Z,    the atomic numbers of the atoms
    * Lmax, real space cutoff
    * Gmax, reciprocal space cutoff
    * a,    a constant to be understood.
    '''
    # The initialization of the Ewald matrix. It is the size corresponding
    # to the largest simulation cells in the data.
    em=np.zeros((80,80))
    # Compute only other triangle later!
    for i in range(N):
        print(i)
        for j in range(i+1,N):
#            if(i==j):
#                em[i,i]=diagonal(Z[i],L,a)
#            else:

            em[i,j]=em[i,j]+short_range(xyz[i,:],xyz[j,:],Z[i],Z[j],L,Lmax,a)

            #                em[i,j]=em[i,j]+long_range(xyz[i,:],xyz[j,:],Z[i],Z[j],L,Gmax,a)
            #                em[i,j]=em[i,j]+self_correction(xyz[i,:],xyz[j,:],Z[i],Z[j],L,a)
            em[j,i]=em[i,j]

    # Arrange the EM according to row norms
    #norms=np.zeros((N,))
    #for i in range(N):
    #    norms[i]=LA.norm(em[i,:])
    #perm=np.invert(np.argsort(norms))
    #em=em[perm,:]
    #em=em[:,perm] # Check this!
                
    return em

def SortRows(M):
    indexlist = np.argsort(np.linalg.norm(M,axis=1))
    M=M[indexlist,:]
    return M[:,indexlist]
    
N=1
data=data.load_data(N)

Lcut=1*np.arange(1,10)
Gcut=4#0.2*np.arange(1,40)
NL=len(Lcut)
Ng=1#len(Gcut)
results=np.zeros((NL,N,80,80))

print(len(data))

for i in range(NL):
    t=time.time()
    for j in range(N):
        a=np.sqrt(np.pi)*(0.01*data[j].N/np.dot(data[j].La[0,:],np.cross(data[j].La[1,:],data[j].La[2,:])))**(1/6)
        results[i,j,:,:]=_C_arraytest.make_ewald_matrix(data[j].N,data[j].La,data[j].xyz,data[j].Z,5.0,Lcut[i],a)
        #results[i,j,:,:]=SortRows(results[0,j,:,:])
    print("L="+str(Lcut[i])+", Time taken : "+str(time.time()-t))

np.save("results2",results)



#453775492
#495998100
#1526246020
#1423127348
