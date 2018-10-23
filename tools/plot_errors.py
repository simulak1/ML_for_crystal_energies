import numpy as np
import matplotlib.pyplot as plt

t=np.load("training_errors.npy")
v=np.load("validation_errors.npy")

fig,ax=plt.subplots(2)
ax[0].plot(t)
ax[1].plot(v)

plt.show()
