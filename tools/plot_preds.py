import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path[0]="/u/82/simulak1/unix/Desktop/doctoral_studies/kurssit/ML_in_MS/src"
import data

preds=np.load("../src/predictions.npy")
materials=data.load_data(np.arange(2400))

targets=np.zeros((300,))

for i in range(300):
    targets[i]=materials[i+2000].Ef

plt.figure(1)
plt.plot(targets,preds,'*')
plt.show()
