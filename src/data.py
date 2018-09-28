import numpy as np
import pandas
import matplotlib.pyplot as plt


class Material:
    def __init__(self,La,xyz,Z,spacegrp,N,p_al,p_ga,p_in,la1_ang,la2_ang,la3_ang,l_angle_alpha_deg,l_ang_beta_deg,l_ang_gamma_deg,Ef,Eb):
        self.N=N
        self.La=La
        self.xyz=xyz
        self.Z=Z
        self.spacegrp=spacegrp
        self.p_al=p_al
        self.p_ga=p_ga
        self.p_in=p_in
        self.la1_ang=la1_ang
        self.la2_ang=la2_ang
        self.la3_ang=la3_ang
        self.l_angle_alpha_deg=l_angle_alpha_deg
        self.l_ang_beta_deg=l_ang_beta_deg
        self.l_ang_gamma_deg=l_ang_gamma_deg
        self.Ef=Ef
        self.Eb=Eb

def load_data(N,datapath="/u/82/simulak1/unix/Desktop/doctoral_studies/kurssit/ML_in_MS/data"):
    ''' 
    Takes in the number of desired datapoints, picks random indices 
    to load data and saves the datapoints into Materials-objects.
    '''
    # Random indices (NOT YET RANDOMIZED)
    indices=np.arange(N)#np.random.randint(N,size=N)
    materials=[]
    rawdata=pandas.read_csv(datapath+"/train.csv",header=None)
    for i in indices:
        La=np.zeros((3,3))
        with open(datapath+"/train/"+str(i+1)+"/geometry.xyz") as f:
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
        materials.append(Material(La,xyz,Z,rawdata[1][i+1],rawdata[2][i+1],rawdata[3][i+1],rawdata[4][i+1],rawdata[5][i+1],rawdata[6][i+1],rawdata[7][i+1],rawdata[8][i+1],rawdata[9][i+1],rawdata[10][i+1],rawdata[11][i+1],rawdata[12][i+1],rawdata[13][i+1]))
    return(materials)
A=load_data(12)
print(A[11].xyz)
