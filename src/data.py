import numpy as np
import pandas
import matplotlib.pyplot as plt


class Material:
    def __init__(self,La,xyz,Z,spacegrp,N,p_al,p_ga,p_in,la1_ang,la2_ang,la3_ang,l_angle_alpha_deg,l_ang_beta_deg,l_ang_gamma_deg,Ef,Eb):
        self.N=int(N)
        self.La=La
        self.xyz=xyz
        self.Z=Z
        self.spacegrp=int(spacegrp)
        self.p_al=p_al
        self.p_ga=p_ga
        self.p_in=p_in
        self.la1_ang=la1_ang
        self.la2_ang=la2_ang
        self.la3_ang=la3_ang
        self.l_angle_alpha_deg=l_angle_alpha_deg
        self.l_angle_beta_deg=l_ang_beta_deg
        self.l_angle_gamma_deg=l_ang_gamma_deg
        self.Ef=Ef
        self.Eb=Eb

def parseGeometryFile(file):
    La=np.zeros((3,3))
    with open(file) as f:
        lines=f.readlines()
        Nlines=len(lines)
        Natoms=Nlines-6
        # Read lattice vectors
        for j in range(3,6):
            line=lines[j].split()
            for k in range(1,4):
                La[j-3,k-1]=float(line[k])
        # Atom coordinates and atomic numbers
        Z=np.zeros((Natoms,))
        xyz=np.zeros((Natoms,3))
        for j in range(6,Nlines):
            line=lines[j].split()
            xyz[j-6,0]=float(line[1])
            xyz[j-6,1]=float(line[2])
            xyz[j-6,2]=float(line[3])
            if(line[4]=="Ga"):
                Z[j-6]=31
            elif(line[4]=="Al"):
                Z[j-6]=13
            elif(line[4]=="O"):
                Z[j-6]=8
            elif(line[4]=="In"):
                Z[j-6]=49
        return La,xyz,Z
                
def load_data(indices,datapath):
    ''' 
    Takes in the number of desired datapoints, picks random indices 
    to load data and saves the datapoints into Materials-objects.
    '''

    materials=[]
    data=np.load(datapath+"/data/data.npy")
    for i in indices:
        if(i<2400):
            file=datapath+"/data/train/"+str(i+1)+"/geometry.xyz"
        else:
            file=datapath+"/data/test/"+str(i+1-2400)+"/geometry.xyz"
        La,xyz,Z=parseGeometryFile(file)
        materials.append(Material(La,xyz,Z,data[i,1],data[i,2],data[i,3],data[i,4],data[i,5],data[i,6],data[i,7],data[i,8],data[i,9],data[i,10],data[i,11],data[i,12],data[i,13]))
    return materials
