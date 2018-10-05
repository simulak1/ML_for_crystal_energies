import numpy as np
import matplotlib.pyplot as plt

m=np.load('results.npy')
m2=np.load('results2.npy')

M=m[0,2,:,:]-m2[0,2,:,:]
plt.imshow(M)
plt.colorbar()
plt.show()
#ind1=np.random.randint(80,size=10)
#ind2=np.random.randint(80,size=10)

#xx=10*np.arange(1,10)

#fig, ((ax1,ax2),(ax3,ax4))=plt.subplots(2,2)


#for i in range(10):
#    ax1.plot(xx,m[:,0,ind1[i],ind2[i]])
#for i in range(10):
#    ax2.plot(xx,m[:,1,ind1[i],ind2[i]])
#for i in range(10):
#    ax3.plot(xx,m[:,2,ind1[i],ind2[i]])
#for i in range(10):
#    ax4.plot(xx,m[:,3,ind1[i],ind2[i]])
#plt.show()

#plt.figure(1)
#plt.imshow(m[0,0,:,:])
#plt.colorbar()
#plt.show()
