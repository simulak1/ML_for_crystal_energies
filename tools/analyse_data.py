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
    
    

data=data.load_data(2400)

Ef=np.zeros((2400,))
Eg=np.zeros((2400,))
N=np.zeros((2400,))
for i in range(2400):
    Ef[i]=data[i].Ef
    Eg[i]=data[i].Eb
    N[i]=data[i].N

energies_hist(Ef,Eg)
atom_number_hist(N)
plt.show()
