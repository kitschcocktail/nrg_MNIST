# -*- coding: utf-8 -*-
"""
Created on Mon Sep 07 10:42:06 2015

@author: heiligenstein
"""
import os

import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from rr import classify_r


def stats(classn, FE_pn, true_y):
    def err(classes_a, classes_b):
        same = []
        for y_a, y_b in zip(classes_a, classes_b):
            same.append((y_a == y_b).astype(float))
        pc_same = 1 - numpy.mean(same)
        return pc_same
        
    print 'error', err(classn, true_y)
    print 'max', numpy.max(FE_pn, axis=0)
    print 'mean', numpy.mean(FE_pn, axis=0)
    
def err(classes_a, classes_b):
        same = []
        for y_a, y_b in zip(classes_a, classes_b):
            same.append((y_a == y_b).astype(float))
        pc_same = 1 - numpy.mean(same)
        return pc_same
    
def plots(datasets, percentages):

    test_x, test_y = datasets
    
    def confusion_matrices(pred_ys, true_ys, pc_cm, save=False):
        cols =  4
        rows = 3
        
        w = (cols * 4)
        h = (rows * 4)
        
        pos = 1
        
        fig = plt.figure(figsize=(w,h))
        for i in xrange(len(pc_cm)):
            cm = confusion_matrix(true_ys, pred_ys[i])
            ax = fig.add_subplot(rows, cols, pos)
            plt.imshow(cm, interpolation="none")
            title = str(pc_cm[i]) + '% imputation; ' + str(numpy.round(err(pred_ys[i], true_ys), 2)) + '% error'
            ax.set_title(title)
            pos += 1
        
        if save:
            os.chdir(os.getcwd() + "/test_results")
            save_name = 'confusion_matrices'
            fig.savefig(save_name, bbox_inches='tight')
            os.chdir('../')
        else:
            plt.show()
                

        
    def just_nrgs(energy_set, pc_cm, save=False):
        nrg, ylabel = energy_set
        fig = plt.figure(figsize=(8, 4))
        plt.boxplot(nrg)
        plt.title(ylabel + ' by % of imputation')
        plt.ylabel(ylabel)
        labels = [(str(x) + '%') for x in pc_cm]
        plt.xticks(range(1, 12), labels)
        if save:
            os.chdir(os.getcwd() + "/test_results")
            fig.savefig('nrgs_bp', bbox_inches='tight')
            os.chdir('../')
        else:
            plt.tight_layout()

    
        
    def energyBP(energy_set, save=False):
        '''Creates boxplots with the calculated energies'''
        impz, impz_rr, impz_r, ylabel = energy_set
        FE_max = numpy.round(numpy.amax(impz[0]), 2)
        cols = len(energy_set) - 1
        rows = 1
        w = (cols * 4)
        h = (rows * 4)
        
        fig, (ax, ax_rr, ax_r) = plt.subplots(rows, cols, figsize=(w,h))
        plt.subplots_adjust(wspace=0.01)
        ax.boxplot(impz)
        ax.set_title('no reconstruction')
        ax.set_xlabel('% imputation')
        ax.set_ylabel(ylabel)
        ax.xaxis.set_ticklabels(percentages)
        ax.text(0.7, -200, str(FE_max))
        for pc_set in impz:
            ax.axhline(y=numpy.median(pc_set), c='0.25', linestyle=':')
        ax_rr.boxplot(impz_rr)
        ax_rr.set_title('reconstruction')
        ax_rr.set_xlabel('% imputation')
        ax_rr.yaxis.set_ticklabels([])
        ax_rr.xaxis.set_ticklabels(percentages)
        for pc_set in impz_rr:
            ax_rr.axhline(y=numpy.median(pc_set), c='0.25', linestyle=':')
        ax_r.boxplot(impz_r)
        ax_r.set_title('reclassification')
        ax_r.set_xlabel('% imputation')
        ax_r.xaxis.set_ticklabels(percentages)
        ax_r.yaxis.set_ticklabels([])
        for pc_set in impz_r:
            ax_r.axhline(y=numpy.median(pc_set), c='0.25', linestyle=':')
        
        if save:
            os.chdir(os.getcwd() + "/test_results")
            save_name = 'FE_boxplots'
            fig.savefig(save_name, bbox_inches='tight')
            os.chdir('../')
        else:
            plt.show()


    def errorBP(error_set, save=False):
        '''Creates boxplots with the calculated errors.'''
        impz, impz_rr, impz_r,= error_set
        
        fig = plt.figure(figsize=(6,4))
        plt.plot(percentages, impz, 'b-', label='No reclassification')
        plt.plot(percentages, impz, 'bo')
        plt.plot(percentages, impz_rr, 'g-', label='Reclass + Reconstr.')
        plt.plot(percentages, impz_rr, 'go')
        plt.plot(percentages, impz_r, 'r-', label='Reclassification')
        plt.plot(percentages, impz_r, 'ro')
        plt.title('Classification Error')
        plt.xticks(range(0, 11))
        plt.xlabel('% imputation')
        plt.ylabel('error rate')
        plt.legend(loc=0)


        if save:
            os.chdir(os.getcwd() + "/test_results")
            save_name = 'err_plot'
            fig.savefig(save_name, bbox_inches='tight')
            os.chdir('../')
        else:
            plt.show()
    
    #-----------------------------------------------------------------
    # for preliminary classification graphs
    pc_cm = range(0, 11)
    true_ys = numpy.array(test_y[0:2000].eval())
    pred_ys = []
    FEs_010 = []
    for pc in pc_cm:
        pred_y_pc = [] ; FEs_pc = []
        for i in xrange(2000):
            print i
            dclass, FE = classify_r(test_x.get_value(borrow=True)[i],
                               imp_pc=pc)[0:2]
            pred_y_pc.append(dclass)
            FEs_pc.append(FE)
        pred_ys.append(pred_y_pc)
        FEs_010.append(FEs_pc)
    
    FE_set = [FEs_010, 'free energy']
    
    confusion_matrices(pred_ys, true_ys, pc_cm, save=True)
    just_nrgs(FE_set, pc_cm, save=True)
    
    #-----------------------------------------------------------------

    errs = []    ; FEs = []
    errs_r = []  ; FEs_r = []
    errs_rr = [] ; FEs_rr = []
    
    for pc in percentages:
        class_pc = []    ; FE_pc = []
        class_pc_r = []  ; FE_pc_r = []
        class_pc_rr = [] ; FE_pc_rr = []
        for i in xrange(2000):
            print i
            # no reclassification
            #dclass, FE, dgt_r = classify_r(test_x.get_value(borrow=True)[i],
            #                               imp_pc=pc)
            y = test_y[i].eval()
            #class_pc.append((dclass == y).astype(float))
            #FE_pc.append(FE)
            # with reclassification
            dclass_r, FE_r, dgt_r_r = classify_r(test_x.get_value(borrow=True)[i],
                                                 reclass=True, imp_pc=pc)
            class_pc_r.append((dclass_r == y).astype(float))
            FE_pc_r.append(FE_r)
            # classification with reconstructed images
            dclass_rr, FE_rr, dgt_r_rr = classify_r(dgt_r_r)
            class_pc_rr.append((dclass_rr == y).astype(float))
            FE_pc_rr.append(FE_rr)
        errs.append(class_pc)       ; FEs.append(FE_pc)
        errs_r.append(class_pc_r)   ; FEs_r.append(FE_pc_r)
        errs_rr.append(class_pc_rr) ; FEs_rr.append(FE_pc_rr)
    

        
            
    
    energy_set = [FEs, FEs_rr, FEs_r, 'free energy']
    
    error_set = [(1-numpy.mean(errs, axis=1)),
                 (1 - numpy.mean(errs_rr, axis=1)),
                 (1 - numpy.mean(errs_r, axis=1))]
    
    energyBP(energy_set, save=True)
    errorBP(error_set, save=True)
