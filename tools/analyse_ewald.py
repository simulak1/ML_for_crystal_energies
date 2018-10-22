import numpy as np
import matplotlib.pyplot as plt

N=2400

data = np.zeros((N-1,80,80))

for i in range(1,N):
    data[i-1,:]=np.load("../data/ewald_matrices/"+str(i)+".npy")
print("Ewald matrices loaded")

maxima=np.zeros((N-1,))
minima=np.zeros((N-1,))
averages=np.zeros((N-1,))

for i in range(N-1):
    maxima[i]=np.max(data[i,:,:])
    minima[i]=np.min(data[i,:,:])
    averages[i]=np.mean(data[i,:,:])


plt.figure(1)
plt.hist(maxima)
plt.title("Maxima")

plt.figure(2)
plt.hist(minima)
plt.title("Minima")

plt.figure(3)
plt.hist(averages)
plt.title("Averages")

plt.show()
