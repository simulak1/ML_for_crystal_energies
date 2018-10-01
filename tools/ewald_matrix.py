'''
This module is for the construction of Ewald matrices.
The matrices are computed based on the atom coordinates 
and atomic numbers (nuclear charges) of the crystal data.
The calculation is performed as in the article of this link:

https://onlinelibrary.wiley.com/doi/full/10.1002/qua.24917

'''
#import scipy
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import scipy as sp
import scipy.special
import sys
import time
#import multiprocessing as mp
from multiprocessing import Pool
from Poolable import make_applicable, make_mappable
from joblib import Parallel, delayed
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

    cosxy=np.dot(Lx,Ly)/LA.norm(Lx)/LA.norm(Ly)
    angxy=np.arccos(cosxy)

    # The maximum distance at each of the lattice vector directions.
    # This method is a guess to include all the lattice vectors, meaning
    # that the number of images considered is made ridicously high. Think of
    # a more elegant method.
    if(LA.norm(Lx)>Lmax or LA.norm(Ly)>Lmax or LA.norm(Lz)>Lmax):sys.exit("ERROR. SHORT_RANGE: The cutoff is too small.")
    Nx=int((Lmax-Lmax%LA.norm(Lx))/LA.norm(Lx))+1
    
    count=0
    count2=0
    # The summing over periodic images
    for i in range(Nx):
        # The following lines aim to minimize the necessary loops over lattice images.
        # a is the fraction of the current Lx-coord to the Lmax, i.e. a=i/Nx*Lmax.
        # beta is the complement angle between the Lx and Ly.
        # ds is the distance from the current Lx coordinate to the edge of the sphere
        # of radius Lmax along the direction of Ly, calculated with cosine formula.
        beta=2*np.pi-angxy
        ds=LA.norm(i*Lx)*np.cos(angxy)+np.sqrt((LA.norm(i*Lx))**2*np.sin(angxy)**2+Lmax**2)
        Ny=int((ds-ds%LA.norm(Ly))/LA.norm(Ly)+1)
        for j in range(Ny):
            if(i>0 and j>0):
                Rxy=i*Lx+j*Ly
                cosxyz=np.dot(Rxy,Lz)/LA.norm(Rxy)/LA.norm(Lz)
                angxyz=np.arccos(cosxyz)
                beta=2*np.pi-angxyz
                ds=LA.norm(Rxy)*np.cos(angxyz)+np.sqrt((LA.norm(Rxy))**2*np.sin(angxyz)**2+Lmax**2)
                Nz=int((ds-ds%LA.norm(Lz))/LA.norm(Lz)+1)
            else:
                Nz=int((Lmax-Lmax%LA.norm(Lz))/LA.norm(Lz))+1
            for k in range(Nz):
                count2=count2+1
                Limage=i*Lx+j*Ly+k*Lz
                if(LA.norm(Limage)<Lmax): # If inside cutoff
                    count=count+1
                    xij=xij+scipy.special.erfc(a*LA.norm(xi-xj+Limage))/LA.norm(xi-xj+Limage)
                    
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
    norms=np.zeros((N,))
    for i in range(N):
        norms[i]=LA.norm(em[i,:])
    perm=np.invert(np.argsort(norms))
    em=em[perm,:]
    em=em[:,perm] # Check this!
                
    return em

N=1
data=data.load_data(N)

#num_cores = mp.cpu_count()

#num_processes_per_node=int((N-N%num_cores)/num_cores)
#node_indices=[]
#for i in range(num_cores-1):
#    node_indices.append(range(i*num_processes_per_node,(i+1)*num_processes_per_node))
#node_indices.append(range((num_cores-1)*num_processes_per_node,N))



a=np.sqrt(np.pi)*(0.01*data[0].N/np.dot(data[0].La[0,:],np.cross(data[0].La[1,:],data[0].La[2,:])))
t=time.time()
#pool=Pool(processes=num_cores)                    
m=make_ewald_matrix(data[0].N,data[0].La,data[0].xyz[0,:],data[0].Z[i],50,10,a)
m=np.zeros((N,N))
#for i in core_indices))
#for i in core_indices:
#    for j in i:
#        for k in i:
#            em[j,k]=
            
print("Time taken: "+str(time.time()-t))
np.save('M1',m)

