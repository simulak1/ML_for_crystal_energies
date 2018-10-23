import numpy as np
import theano
import theano.tensor as T
import data
import sys

def load_ewald(target):

    # Load labels
    materials=data.load_data(np.arange(2400))

    # Load all of the ewald matrices
    Xdata=np.zeros((2400,80,80))
    for i in range(2400):
        Xdata[i,:,:]=np.load('../data/ewald_matrices/'+str(i)+".npy")

    # Reshape and normalize the data
    #print(np.max(Xdata))
    Xdata=np.negative(Xdata)
    Xdata=Xdata/np.max(Xdata)
    Xdata=Xdata.reshape(-1,1,80,80)

    # Split the data into training, validation and test sets
    Xtrain,Xval,Xtest=Xdata[:1800],Xdata[1800:2200],Xdata[2200:2400]

    Ydata=np.zeros((2400,))
    # Load targets
    for i in range(2400):
        if target=='Ef':
            Ydata[i]=100*materials[i].Ef
        elif target=='Eb':
            Ydata[i]=materials[i].Eb
        else:
            sys.exit("Error: unknown target property.")

    # Split the targets
    Ytrain,Yval,Ytest=Ydata[:1800],Ydata[1800:2200],Ydata[2200:2400]

    return Xtrain,Ytrain,Xval,Yval,Xtest,Ytest
        
        
# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

