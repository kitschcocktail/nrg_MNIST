# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 18:07:01 2015

@author: heiligenstein
"""
import os
import numpy
import time

import cPickle
from logistic_sgd import load_data

tot_start_time = time.clock()

dataset = 'mnist.pkl.gz'
datasets = load_data(dataset)
test_x, test_y = datasets[2]


# import parameters
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

digit_weights = numpy.loadtxt('digit_weights.txt')
inv_weights = 1 - digit_weights

os.chdir("..") 

def tenk(reclass_iters, imp_pc, c_ones):
    
    reclass_iters, imp_pc, c_ones = int(reclass_iters), int(imp_pc), int(c_ones)
    
    labels = []
    for i in range(0, 10):
        label = numpy.zeros(10)
        label[i] = 1
        labels.append(label)
    labels = numpy.array(labels)
    
    def ran_c1(input_array, pc, c_ones):
        """Ramdomly inserts consecutive 1s."""
        impc = numpy.array(input_array)
        length = len(input_array)
        num_zeros = int(length * float(pc) / 100)
        num_outs = int(num_zeros/c_ones)
        remaining_zeros = (num_zeros)%(c_ones)
        for i in range(num_outs):
            rand = numpy.random.randint(0, high=(length - c_ones))
            j = c_ones
            while j != 0:
                if impc[rand%length] != 1:
                    impc[rand%length] = 1
                    j -= 1
                rand += 1
        if remaining_zeros != 0:
            rand2 = numpy.random.randint(0, high=(length - remaining_zeros))
            j2 = remaining_zeros
            while j2 != 0:
                if impc[rand2%length] != 1:
                    impc[rand2%length] = 1
                    j2 -= 1
                rand2 +=1
        return impc
        
    def fe_v(vlayer, vbias, hbias, W): # works
        ''' Function to compute free energy of visible vector v'''
        # f_e = -sumVisibleBias - sumLog(1 + exp(sumVisibleWeight + hbias))
        sum_vb = numpy.dot(vlayer, vbias)
        vWb = (numpy.dot(vlayer, W)) + hbias
        sumLog = sum(numpy.log(1 + numpy.exp(vWb))) 
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
        
    def classify(digit_image, wts=False):
        # Perform a bottom-up pass to get wake/positive phase probabilites 
        # and sample states
        fe0            = fe_v(digit_image, vbias0, hbias0, W0)        
        wakehidprobs   = sigmoid(numpy.dot(digit_image, W0) + hbias0)
        wakehidstates  = (wakehidprobs > numpy.random.rand(1, 500)).astype(float)
        fe1            = fe_v(wakehidstates[0], vbias1, hbias1, W1)
        wakepenprobs   = sigmoid(numpy.dot(wakehidstates ,W1) + hbias1)
        wakepenstates  = (wakepenprobs > numpy.random.rand(1, 500)).astype(float)
        
        waketopprobs   = sigmoid(numpy.dot(wakepenstates, W2) + hbias2)
        waketopstates  = (waketopprobs > numpy.random.rand(1,2000)).astype(float)
    
        clayerstates = (numpy.dot(waketopstates[0], U.T) + dbias)
        classified_as = numpy.where(clayerstates == numpy.amax(clayerstates))[0][0]        
        class_label = labels[classified_as]
        fec = fe_vc(wakepenstates[0], hbias2, W2, class_label, dbias, U)
        
        FE_vc = fe0 + fe1 + fec # total network free energy
        
        if wts:
            out = classified_as, FE_vc, waketopstates[0]
        else:
            out = classified_as, FE_vc
        return out
        
    def generate(waketopstates, dclass, numCDiters):
    # Perform numCDiters Gibbs Sampling iterations using the top level
    # undirected associative memory
        class_label = labels[dclass]
        negtopstates = waketopstates # to initialize the loop
        for k in xrange(numCDiters):
            negpenprobs  = sigmoid(numpy.dot(negtopstates, W2.T) + vbias2)
            negpenstates = (negpenprobs > numpy.random.rand(1,500)).astype(float)
            neglabprobs  = class_label # label clamped
            negtopprobs  = sigmoid(numpy.dot(negpenstates, W2) + numpy.dot(neglabprobs,U) + hbias2)
            negtopstates = (negtopprobs > numpy.random.rand(1,2000)).astype(float)
    
        sleeppenstates = negpenstates
        sleephidprobs = sigmoid(numpy.dot(sleeppenstates, W1.T) + vbias1)
        sleephidstates = (sleephidprobs > numpy.random.rand(1,500)).astype(float)
        sleepvisprobs  = sigmoid(numpy.dot(sleephidstates, W0.T) + vbias0)
    
        return sleepvisprobs[0]
    
    def classify_r(digit_image, reclass_iters):
        
        threshold = -258 # this should be part of the parameters when retrained
        CDiters = 500 # contrastive divergence iterations
        
        # perform classification, and retrieve top level for generation
        dclass, FE, wts = classify(digit_image, wts=True)
        
        # if reclassification is enabled, reclassify only when FE is above th
        if FE > threshold:
            #print '   FE over threshold. Reclassifying', reclass_iters, 'times.'
            cl_r = numpy.zeros(10)
            FE_r = []
            wts_r = numpy.zeros_like(wts)
            for i in xrange(reclass_iters):
                #print '             ', i
                mclass, mFE, mwts = classify(digit_image, wts=True)
                mcl = labels[mclass]
                cl_r += mcl
                FE_r.append(mFE)
                wts_r += mwts
            #class is chosen based on greatest count
            dclass_r = numpy.where(cl_r == numpy.amax(cl_r))[0][0]
            FE_r = numpy.mean(FE_r)
            wts_r = numpy.round(wts_r / reclass_iters)
            # use chosen class to generate new digit, and mix with original
            # corrupted image
            gen_dgt_img = generate(wts_r, dclass_r, CDiters)
            weighted_noise = digit_weights * digit_image
            weighted_gen = inv_weights * gen_dgt_img
            mix = weighted_noise + weighted_gen

            out = dclass_r, FE_r, mix
        else:
            out = dclass, FE, digit_image
        
        return out
    

    # corruption, no reclass, no reconstruction
    if reclass_iters == 0:
        print 'No reclassification'
        FEs_impc = [] ; errs_impc = []
        counter = 0
        for dgt_img, y in zip(test_x.get_value(borrow=True), test_y.eval()):
        #for i in xrange(100):
            if counter%100 == 0:
                print 'input vector', counter
           
            corr_dgt_img = ran_c1(dgt_img, imp_pc, c_ones)
        
            dclass_c, FE_c = classify(corr_dgt_img)
            errs_impc.append((dclass_c == y).astype(float))
            FEs_impc.append(FE_c)
            counter += 1
        out_file = [FEs_impc, errs_impc]
        

    # with reclassification
    else:
        print 'Reclassification. n =', reclass_iters
        FEs_rr_impc = [] ; errs_rr_impc = []
        FEs_r_impc = [] ; errs_r_impc = [] 
        
        counter = 0
        
        for dgt_img, y in zip(test_x.get_value(borrow=True), test_y.eval()):
        #for i in xrange(100):
            if counter%100 == 0:
                print 'input vector', counter
           
            corr_dgt_img = ran_c1(dgt_img, imp_pc, c_ones)

            dclass_r_c, FE_r_c, dgt_r_c = classify_r(corr_dgt_img,
                                                     reclass_iters=reclass_iters)
            errs_r_impc.append((dclass_r_c == y).astype(float))
            FEs_r_impc.append(FE_r_c)
            # classification with reconstructed images
            dclass_rr_c, FE_rr_c = classify(dgt_r_c)
            errs_rr_impc.append((dclass_rr_c == y).astype(float))
            FEs_rr_impc.append(FE_rr_c)
            counter += 1

        out_file = [FEs_rr_impc, errs_rr_impc, 
                    FEs_r_impc, errs_r_impc]

    
    save_name = 'errfe'+ str(reclass_iters)+ '_' +\
                str(imp_pc)+ '_' + str(c_ones) + '.txt'
    
    os.chdir("test_results")
    numpy.savetxt(str(save_name), out_file, fmt='%1.2e')
    print 'Results saved in /test_results as', save_name 
    os.chdir("..")
    
    
    if imp_pc:
        print 'Data corrupted at', imp_pc, '%, with length', c_ones
    else: 
        print 'No corruption.'
    
    tot_end_time = time.clock()
    print 'Classification ran for %.2fm' % ((tot_end_time - tot_start_time) / 60.)


if __name__ == "__main__":
    import sys
    tenk(sys.argv[1], sys.argv[2], sys.argv[3])
