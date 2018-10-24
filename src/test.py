from __future__ import print_function
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

def imgplot(params,layer,Niter,Nchannel,xcol,ycol):
    plt.ion()
    fig,ax=plt.subplots(xcol,ycol)
    for i in range(Niter):
        ch=0
        for j in range(xcol):
            for k in range(ycol):
                ax[j,k].clear()
                if(j==0 and k==0):ax[0,0].set_title("Iteration step: "+str(i))
                ax[j,k].imshow(params[i][layer][ch][0])
                ch=ch+1
        fig.canvas.draw()
        plt.pause(.1)

def lineplot(params,N,layer,Niter,Nchannels):
    x=params[0][layer].flatten(0)
    Nparams=len(x)
    RandomIndices=np.random.randint(Nparams,size=N)

    weights=np.zeros((Niter,N))
    biases=np.zeros((Niter,Nchannels))
    for i in range(Niter):
        x=params[i][layer].flatten(0)
        weights[i,:]=x[RandomIndices]
        biases[i,:]=params[i][layer+1]

    fig,ax=plt.subplots(2)
    for i in range(N):
        ax[0].plot(weights[:,i])
    for i in range(Nchannels):
        ax[1].plot(biases[:,i])
    plt.show()
        
def main(layer=0, N_samples=10, plt_type='lineplot',speed=1,xcol=1,ycol=1):
    # Load parameters
    with np.load('params.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    print("The shape of the network layers are:")
    for i in range(len(param_values[0])):
        print("layer "+str(i+1)+" : "+ str(param_values[0][i].shape))
    
    Niter=len(param_values)

    LayerIndex=2*layer
    if len(param_values[0][LayerIndex].shape)<3:
        sys.exit("The chosen layer does not appear to be convolutional. Use the script for plotting parameters of fully connected layers for this layer.")

    NChannels=param_values[0][LayerIndex].shape[0]

    if(plt_type=='lineplot'):
        lineplot(param_values,N_samples,LayerIndex,Niter,NChannels)
    elif(plt_type=='imgplot'):
        imgplot(param_values,LayerIndex,Niter,NChannels,xcol,ycol)
        
if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("A tool for plotting weights and biases of a convolutional neural network.")
        print("Syntax: python mnist.py [Index of layer] [Number of random samples] [line plot or matrix image] [shape of the subplot of channel weights]  [speed of image plot]")
        print("Index of layer           : 0 means the input layer, calculate further from that.")
        print("Number of random samples : If line plot is used, you probably want only some random weights for clarity")
        print("plt_type                 : either 'lineplot' or 'image' ")
        print("Speed                    : How many training steps you skip between image plots? You probably have quite many steps.")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['layer'] = int(sys.argv[1])
        if len(sys.argv) > 2:
            kwargs['N_samples'] = int(sys.argv[2])
        if len(sys.argv) > 3:
            kwargs['plt_type'] = sys.argv[3]
        if len(sys.argv) > 5:
            kwargs['xcol'] = int(sys.argv[4])
            kwargs['ycol'] = int(sys.argv[5])
        if len(sys.argv) > 6:
            kwargs['speed'] = sys.argv[4]


main(**kwargs)
