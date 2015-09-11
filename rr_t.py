# -*- coding: utf-8 -*-
"""
Created on Tue Sep 01 14:18:52 2015

@author: heiligenstein
"""

import os

import numpy
import random

import cPickle
import matplotlib.pyplot as plt
from pylab import cm




def classify_r(digit_image, imp_pc=None):
    
    os.chdir("output_folder")

    f = cPickle.load(open('DBN.pkl','rb'))
    W0     = f[0][0].get_value(borrow=True)
    hbias0 = f[0][1].get_value(borrow=True)
    vbias0 = f[0][2].get_value(borrow=True)
    
    W1     = f[1][0].get_value(borrow=True)
    hbias1 = f[1][1].get_value(borrow=True)
    vbias1 = f[1][2].get_value(borrow=True)
    
    W2     = f[2][0].get_value(borrow=True)
    hbias2 = f[2][1].get_value(borrow=True)
    vbias2 = f[2][2].get_value(borrow=True)
    U      = f[2][3].get_value(borrow=True)
    dbias  = f[2][4].get_value(borrow=True)
    
    os.chdir("..")    

    threshold = -207 # this should be part of the parameters when retrained
    
    def fe_v(vlayer, vbias, hbias, W): # works
        ''' Function to compute free energy of visible vector v'''
        # f_e = -sumVisibleBias - sumLog(1 + exp(sumVisibleWeight + hbias))
        sum_vb = numpy.dot(vlayer, vbias)
        vWb = (numpy.dot(vlayer, W)) + hbias
        sumLog = sum(numpy.log(1 + numpy.exp(vWb))) 
        # numpy.log(1 + numpy.exp(numpy.dot(layer1statessmall,
        #                       numpy.transpose(Wsmall)[0]) + hbias1small[0]))
        F = -sum_vb - sumLog
        return F
        
    def fe_vc(vlayer, hbias, W, clayer, cbias, U):
        ''' Function to compute free energy of visible vector v and class c'''
        # f_e = -sumClassBias - sumLog(1 + exp(sumVisibleWeight + hbias + sumClassWeight))
        sum_cb = numpy.dot(clayer, cbias)
        vWbcU = (numpy.dot(vlayer, W)) + hbias + (numpy.dot(clayer, U))
        sumLog = sum(numpy.log(1 + numpy.exp(vWbcU)))
        F_vc = -sum_cb - sumLog
        return F_vc
    
    def sigmoid(z):
        return  1 / (1 + numpy.exp(-z))
        
    def img(digit_vector):
        plt.imshow(-digit_vector.reshape(28,28), cmap=cm.gray)
    
    def ran_1(input_array, pc):
        """Imputes randomly scattered 1s."""
        x = int(len(input_array) * (float(pc) / 100))
        node = random.sample(xrange(0, len(input_array)), x)
        impo = numpy.array(input_array)
        for i in node:
            impo[i] = 1
            return impo
    
    
    labels = []
    for i in range(0, 10):
        label = numpy.zeros(10)
        label[i] = 1
        labels.append(label)
    labels = numpy.array(labels)
    
    digit_image = ran_1(digit_image, imp_pc)
    
        
    def classify(dgt_img):
        # Perform a bottom-up pass to get wake/positive phase probabilites 
        # and sample states
        wakehidprobs   = sigmoid(numpy.dot(dgt_img,W0) + hbias0)
        wakehidstates  = (wakehidprobs > numpy.random.rand(1, 500)).astype(float)[0]
        fe0            = fe_v(dgt_img, vbias0, hbias0, W0)
        wakepenprobs   = sigmoid(numpy.dot(wakehidstates,W1) + hbias1)
        wakepenstates  = (wakepenprobs > numpy.random.rand(1, 500)).astype(float)[0]
        fe1            = fe_v(wakehidstates, vbias1, hbias1, W1)
        waketopprobs   = sigmoid(numpy.dot(wakepenstates,W2) + hbias2)
        waketopstates  = (waketopprobs > numpy.random.rand(1,2000)).astype(float)[0]
    
        clayerstates = (numpy.dot(waketopstates, U.T) + dbias)
        classified_as = numpy.where(clayerstates == numpy.amax(clayerstates))[0][0]        
        class_label = labels[classified_as]
        fec = fe_vc(wakepenstates, hbias2, W2, class_label, dbias, U)
        
        FE_vc = fe0 + fe1 + fec # total network free energy
        
        # calculate the posterior class probability distribution given
        # visible vector of last RBM
        # p(y=class|x) = sum(p(y=class, h|x))
        #              = e(-F(x, class)) / sum(e(-F(x, class')))
        top = numpy.exp(-FE_vc)
        all_FEs = []
        for label in labels:
            FE_prime = fe_vc(wakepenstates, hbias2, W2, label, dbias, U)
            FE_prime = fe0 + fe1 + FE_prime        
            all_FEs.append(numpy.exp(-FE_prime))
        bottom = sum(all_FEs)
        
        p_yx =  top / bottom
        
        return classified_as, FE_vc, p_yx, waketopstates
    
    dclass, FE, p, wts = classify(digit_image)

    def generate(waketopstates, dclass, numCDiters):
    # Perform numCDiters Gibbs Sampling iterations using the top level
    # undirected associative memory
        class_label = labels[dclass]
        negtopstates = waketopstates # to initialize the loop
        for k in range(numCDiters):
            negpenprobs  = sigmoid(numpy.dot(negtopstates,W2.T) + vbias2)
            negpenstates = (negpenprobs > numpy.random.rand(1,500)).astype(float)
            neglabprobs  = class_label # label clamped
            negtopprobs  = sigmoid(numpy.dot(negpenstates,W2) + numpy.dot(neglabprobs,U) + hbias2)
            negtopstates = (negtopprobs > numpy.random.rand(1,2000)).astype(float)
    
        sleeppenstates = negpenstates
        sleephidprobs = sigmoid(numpy.dot(sleeppenstates,W1.T) + vbias1)
        sleephidstates = (sleephidprobs > numpy.random.rand(1,500)).astype(float)
        sleepvisprobs  = sigmoid(numpy.dot(sleephidstates,W0.T) + vbias0)
    
        return sleepvisprobs[0]
    """
    if th='FE':
        threshold = FE
    elif th='p':
        threshold = p
    else:
        print 'Choose threshold as free energy or probability'"""
    
    if threshold and FE > threshold:
        #print 'Free energy is', FE, '. Reclassifying...'
        count = numpy.zeros(10)
        for i in range(25):
            mclass, FE, p, wts= classify(digit_image)
            mcl = labels[mclass]
            count += mcl
        dclass = numpy.where(count == numpy.amax(count))[0][0]
        recon_digit = generate(wts, dclass, 100)
        
        out_digit_image = recon_digit
        
    return dclass, FE, p, out_digit_image
    
