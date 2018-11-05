'''
This module contains a function to make a matrix of distances
between atoms in a supercell, consisting of the original simulation cell multiplied.
'''

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import scipy as sp
import scipy.special
import sys
import time
sys.path[0]="/u/82/simulak1/unix/Desktop/doctoral_studies/kurssit/ML_in_MS/src"
import data

def writeGeometryIn(L,xyz,Z,name="geometry.in"):
    with open(name,"w+") as f:
        for i in range(3):
            f.write("lattice vector "+ str(L[i,0])+" "+ str(L[i,1])+" "+ str(L[i,2])+"\n")
        for i in range(len(xyz[:,0])):
            if Z[i]==31:
                f.write("atom "+str(xyz[i,0])+" "+str(xyz[i,1])+" "+str(xyz[i,2])+" Ga"+"\n")
            elif Z[i]==13:
                f.write("atom "+str(xyz[i,0])+" "+str(xyz[i,1])+" "+str(xyz[i,2])+" Al"+"\n")
            elif Z[i]==49:
                f.write("atom "+str(xyz[i,0])+" "+str(xyz[i,1])+" "+str(xyz[i,2])+" In"+"\n")
            elif Z[i]==8:
                f.write("atom "+str(xyz[i,0])+" "+str(xyz[i,1])+" "+str(xyz[i,2])+" O"+"\n")

def SuperCell(L,xyz,Z,xm=0,xp=0,ym=0,yp=0,zm=0,zp=0):
    N=len(xyz[:,0])
    Nx=len(range(xm,xp+1))
    Ny=len(range(ym,yp+1))
    Nz=len(range(zm,zp+1))
    xyz_s=np.zeros((Nx*Ny*Nz*N,3))
    ind=0
    for ix in range(xm,xp+1):
        for iy in range(ym,yp+1):
            for iz in range(xm,xp+1):
                for i in range(N):
                    Lx=ix*L[0,:]
                    Ly=iy*L[1,:]
                    Lz=iz*L[2,:]
                    translation=np.sum([Lx,Ly,Lz],axis=0)
                    #            translation=np.sum(Lz,translation,axis=0)
                    xyz_s[ind,:]=np.sum([xyz[i,:],translation],axis=0)
                    ind=ind+1
    return xyz_s

