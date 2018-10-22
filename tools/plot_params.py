from __future__ import print_function
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

def imgplot(l,params):
    Niter=len(params)
    plt.ion()
    fig,ax=plt.subplots(1)
    for i in range(Niter):
        ax.imshow(params[i][l])
        ax.set_title("Epoch: "+str(i))
        fig.canvas.draw()
        plt.pause(0.1)
        ax.clear()
        
def lineplot(l,nodeind, weightind, params):
    Niter=len(params)
    Nsamples=len(nodeind)

    weights=np.zeros((Niter,Nsamples))
    biases=np.zeros((Niter,Nsamples))

    for i in range(Niter):
        for j in range(Nsamples):
            biases[i,j]=params[i][l-1][j]
            weights[i,j]=params[i][l][nodeind[j],weightind[j]]
    
    fig,ax=plt.subplots(2)
    for i in range(Nsamples):
        ax[0].plot(weights[:,i])
        ax[1].plot(biases[:,i])
    plt.show()

def main(layer=0, N_samples=10, plt_type='lineplot',speed=1):
    # Load parameters
    with np.load('params.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    Niter=len(param_values)

    if layer>0:
        LayerWeightIndex=2*layer
        LayerBiasIndex=2*layer-1
    elif layer==0:
        sys.exit("Cannot plot the input layer weight at the moment")

    # This index 1 is not general enough for beautiful code, but works for now
    weights=param_values[1][LayerWeightIndex]
    biases=param_values[1][LayerBiasIndex]
    
    NNodesInLayer=len(biases)
    NWeightsInNode=weights[0].shape[0]

    if plt_type=='lineplot':
        SampleNodeIndices=np.random.randint(NNodesInLayer,size=N_samples)
        SampleWeightIndices=np.random.randint(NWeightsInNode,size=N_samples)
        lineplot(LayerWeightIndex,SampleNodeIndices,SampleWeightIndices,param_values)
    elif plt_type=='imgplot':
        imgplot(LayerWeightIndex,param_values)
        
if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("A tool for plotting weights and biases of a neural network.")
        print("Syntax: python mnist.py [Index of layer] [Number of random samples] [line plot or matrix image] [speed of image plot]")
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
        if len(sys.argv) > 4:
            kwargs['speed'] = sys.argv[4]


main(**kwargs)
