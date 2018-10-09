import numpy as np
import matplotlib.pyplot as plt

m=np.load('results2.npy')
'''
M=m[0,0,:,:]
plt.imshow(M)
plt.colorbar()
plt.show()
'''
ind1=np.random.randint(80,size=20)
ind2=np.random.randint(80,size=20)

xx=np.arange(1,10)

fig=plt.figure(1)


for i in range(10):
    plt.plot(xx,m[:,0,ind1[i],ind2[i]])
plt.title("Long range interaction convergence wit Gmax")
plt.show()

'''for i in range(10):
    ax2.plot(xx,m[:,1,ind1[i],ind2[i]])
for i in range(10):
    ax3.plot(xx,m[:,2,ind1[i],ind2[i]])
for i in range(10):
    ax4.plot(xx,m[:,2,ind1[i],ind2[i]])
plt.show()
#plt.figure(1)
#plt.imshow(m[0,0,:,:])
#plt.colorbar()
#plt.show()

'''
