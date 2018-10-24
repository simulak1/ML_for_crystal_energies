#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.
This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html
More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import datahandle
import neural_network_structures as NNS

import lasagne

# ############################## Main program ################################

def main(model='mlp', num_epochs=500,continuation_run=0,target_property='Ef'):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = datahandle.load_ewald(target_property,datapath)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.dvector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if model == 'mlp':
        network = NNS.build_mlp(input_var)
    elif model.startswith('custom_mlp:'):
        depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')
        network = NNS.build_custom_mlp(input_var, int(depth), int(width),
                                   float(drop_in), float(drop_hid))
    elif model == 'cnn':
        network = NNS.build_cnn(input_var)
    elif model.startswith('custom_cnn:'):
        nconv, fwidth, nch, nfc, width, drop = model.split(':', 1)[1].split(',')
        network = NNS.build_custom_cnn(input_var,int(nconv), int(fwidth), int(nch),
                                       int(nfc), int(width), float(drop))
    else:
        print("Unrecognized model type %r." % model)
        return

    if(continuation_run==1):
        with np.load('model.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)

        
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(
        loss, params, learning_rate=0.002, beta1=0.9, beta2=0.999,epsilon=1e-08)
    #Adam: #            loss, params, learning_rate=0.005, beta1=0.9, beta2=0.999,epsilon=1e-08)
    # Nesterov: # loss, params, learning_rate=0.005, momentum=0.4)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], test_loss)

    # Next, compile a function that returns the predicted values
    pred_fn = theano.function([input_var], prediction)
    ##########################################################
    # Initialize stuff to save for analysis of the algorithm #
    ##########################################################
    allparams=[]                                             #
    isave=int(num_epochs/100)                                #
    if isave==0:isave=1                                      #
    train_errors=np.zeros((num_epochs,))                     #
    val_errors=np.zeros((num_epochs,))                       #
    ##########################################################
    ##########################################################
    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in datahandle.iterate_minibatches(X_train, y_train, 50, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        for batch in datahandle.iterate_minibatches(X_val, y_val, 50, shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, targets)
            val_err += err
            val_batches += 1

        ####################### Save necessary stuff for further analysis #################
        # Save parameters. The wrestling with isave is to ensure we don't run out of memory
        if epoch%isave==0:
            allparams.append([*lasagne.layers.get_all_param_values(network)])
        train_errors[epoch]=train_err/train_batches
        val_errors[epoch]=val_err/val_batches
        ###################################################################################
            
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        
    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in datahandle.iterate_minibatches(X_test, y_test, 50, shuffle=False):
        inputs, targets = batch
        err = val_fn(inputs, targets)
        test_err += err
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))

    np.savez('params.npz', *allparams)
        
    # Optionally, you could now dump the network weights to a file like this:
    np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    # Then save predictions for comparing against true values
    for batch in datahandle.iterate_minibatches(X_test, y_test, 200, shuffle=False):
        inputs, targets = batch
        preds = pred_fn(inputs)
    np.save('predictions',preds)
    np.save('training_errors',train_errors)
    np.save('validation_errors',val_errors)
    

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("       'custom_cnn:NCONVLAYERS,FILTERWIDTH,NCHANNELS,")
        print("                   NFCLAYERS, WIDTH, DROP for a CNN architecture")
        print("                   with NCONVLAYERS convolutional layers with")
        print("                   NCHANNELS x FILTERWIDTH x FILTERWIDTH filters")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        if len(sys.argv) > 3:
            kwargs['continuation_run'] = int(sys.argv[3])
        if len(sys.argv) > 4:
            kwargs['datapath'] = sys.argv[4]
        if len(sys.argv) > 5:
            kwargs['target_property'] = sys.argv[4]
            
            
main(**kwargs)
