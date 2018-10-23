import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path[0]="/u/82/simulak1/unix/Desktop/doctoral_studies/kurssit/ML_in_MS/src"
import data

preds=np.load("../src/predictions.npy")
materials=data.load_data(np.arange(2400))

targets=np.zeros((200,))

for i in range(200):
    targets[i]=100*materials[i+2200].Ef

xx=100*np.arange(0,0.5,0.001)
plt.figure(1)
plt.plot(targets,preds,'*',xx,xx,'r-')

plt.show()
