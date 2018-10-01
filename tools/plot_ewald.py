import numpy as np
import matplotlib.pyplot as plt

m1=np.load('M1.npy')
m2=np.load('M2.npy')
print(m1.shape)
plt.figure(1)
plt.imshow(m1)
plt.colorbar()
plt.figure(2)
plt.imshow(m2)
plt.colorbar()
plt.show()
