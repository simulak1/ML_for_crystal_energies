import sys
sys.path[0]="/u/82/simulak1/unix/Desktop/doctoral_studies/kurssit/ML_in_MS/src"
import data
sys.path="$PWD"
import numpy as np
import matplotlib.pyplot as plt


def energies_hist(Ef,Eg):
    fig,ax=plt.subplots(2)
    ax[0].hist(Ef,200)
    ax[0].set_title("Distribution of formation energies (eV)")
    ax[1].set_title("Distribution of band gap energies  (eV)")
    ax[1].hist(Eg,200)
    

def atom_number_hist(N):
    plt.figure(11)
    plt.hist(N,8)
    plt.title("Distribution of atom numbers in a simulation cell")
    
    
def angles_hist(a,b,c):
    fig,ax=plt.subplots(3)
    ax[0].hist(a,200)
    ax[0].set_title("Distribution of alpha angles")
    ax[1].set_title("Distribution of beta angles")
    ax[1].hist(b,200)
    ax[2].hist(c,200)
    ax[2].set_title("Distribution gamma angles")
                        
                            
    
    
data=data.load_data(2400)

Ef=np.zeros((2400,))
Eg=np.zeros((2400,))
N=np.zeros((2400,))
alpha=np.zeros((2400,))
beta=np.zeros((2400,))
gamma=np.zeros((2400,))

for i in range(2400):
    Ef[i]=data[i].Ef
    Eg[i]=data[i].Eb
    N[i]=data[i].N
    alpha[i]=data[i].l_angle_alpha_deg
    beta[i]=data[i].l_angle_beta_deg
    gamma[i]=data[i].l_angle_gamma_deg
    
energies_hist(Ef,Eg)
atom_number_hist(N)
angles_hist(alpha,beta,gamma)
plt.show()
