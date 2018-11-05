import numpy as np
import theano
import theano.tensor as T
import data
import sys

def load_ewald(target,datapath,round):
    # Currently training set has 2400 and validation set has 400
    # elements, so there are 7 ways to choose the partitioning
    assert(round<7)
    # Load labels
    materials=data.load_data(np.arange(3000),datapath)

    # Load all of the ewald matrices
    Xdata=np.zeros((3000,80,80))
    for i in range(3000):
        Xdata[i,:,:]=np.load(datapath+'/data/ewald_matrices/'+str(i)+".npy")

    # Reshape and normalize the data
    #print(np.max(Xdata))
    Xdata=np.float32(Xdata)
    Xdata=np.negative(Xdata)
    Xdata=Xdata/np.max(Xdata)
    Xdata=Xdata.reshape(-1,1,80,80)

    Xtrain=np.zeros((2400,1,80,80))
    Xval=np.zeros((400,1,80,80))
    Xtest=np.zeros((200,1,80,80))
    Ytrain=np.zeros((2400,))
    Yval=np.zeros((400,))
    Ytest=np.zeros((200,))
    print(round)
    # Split the data into training, validation and test sets
    Xtrain[:round*400]=Xdata[:round*400]
    Xtrain[round*400:]=Xdata[(round+1)*400:2800]
    Xval=Xdata[round*400:(round+1)*400]
    Xtest=Xdata[2800:3000]

    Ydata=np.zeros((3000,))
    # Load targets
    for i in range(3000):
        if target=='Ef':
            Ydata[i]=100*materials[i].Ef
        elif target=='Eb':
            Ydata[i]=materials[i].Eb
        else:
            sys.exit("Error: unknown target property.")

    # Split the targets
    Ytrain[:round*400]=Ydata[:round*400]
    Ytrain[round*400:]=Ydata[(round+1)*400:2800]
    Yval=Ydata[round*400:(round+1)*400]
    Ytest=Ydata[2800:3000]
    
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

def iterate_minibatches(inputs, targets, batchsize, shuffle=False, augment=False,aug=0.0):
    assert len(inputs) == len(targets)
    outputs=np.zeros(inputs.shape,dtype=np.float32)
    if augment:
        for i in range(len(inputs)):
            norms=np.linalg.norm(inputs[i],axis=1)
            noise=np.random.normal(0.0,aug,size=80)
            newnorms=np.sum([norms,noise],axis=0)
            indexlist=np.argsort(newnorms)
            outputs[i]=inputs[i,0,indexlist,:]
            outputs[i]=outputs[i,0,:,indexlist]
    else:
        outputs=inputs
    outputs=np.float32(outputs)
    if shuffle:
        indices = np.arange(len(outputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(outputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield outputs[excerpt], targets[excerpt]

