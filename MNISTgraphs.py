# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 15:17:01 2015

@author: heiligenstein
"""
import os
import numpy
import matplotlib.pyplot as plt
from pylab import cm
from glob import glob

ns = [0, 25, 50, 100, 1000]
percentages = [0,1,2,5,10]
lengths = [1, 2, 5, 10]

def move_ten(fname_list):
    ten = fname_list[1]
    fname_list.remove(ten)
    fname_list.append(ten)
    return fname_list

def nrgserrs_0n(save=False):
    fnames_0 = glob('errfe0_*1.txt')
    fnames_0 = move_ten(fnames_0)


    out_file = []
    control_FEs = []
    pc = 0
    for control_set in fnames_0:
        FE, err = numpy.loadtxt(control_set)
        control_FEs.append(FE)
        out_file.append([pc,
                         numpy.mean(FE),
                         1 - numpy.mean(err)])
        pc +=1

    fig = plt.figure(figsize=(8, 4))
    plt.boxplot(control_FEs)
    plt.title('F by % of corruption')
    plt.ylabel('free energy')
    labels = [(str(x) + '%') for x in range(0, 11)]
    plt.xticks(range(1, 12), labels)
    if save:
        fig.savefig('FE_boxplots0n1c', bbox_inches='tight')
        save_name = 'errfe_stats_0n.csv'
        header = 'pc,Fmean,er'
        numpy.savetxt(save_name, out_file, delimiter=',',
                      fmt='%1.0f,%1.2f,%1.2f',
                      header=header, comments='')
    else:
        plt.tight_layout()
        print numpy.array(out_file)
        
        
def stats_pc(c, save=False):
    '''Prints/Saves a file for each corruption %. Each file contains
    4 columns: mean FE rr, err rr, mean FE r, err r
    5 rows: header, n = 0, 25, 50, 100, 1000 '''
    
    if c == 0:
        out_file = []
        fnames_0 = glob('errfe*_0_1.txt')
        hund = fnames_0[2]
        fnames_0.remove(hund)
        fnames_0.append(hund)
        thou = fnames_0[1]
        fnames_0.remove(thou)
        fnames_0.append(thou)
        FE_0n, err_0n = numpy.loadtxt(fnames_0[0])
        out_file.append([0, 0, numpy.mean(FE_0n), 1 - numpy.mean(err_0n), numpy.mean(FE_0n), 1 - numpy.mean(err_0n)])
        for fname, n in zip(fnames_0[1:], [25,50,100,1000]):
            FE_rr, err_rr, FE_r, err_r = numpy.loadtxt(fname)
            out_file.append([0,
                             n,
                             numpy.mean(FE_rr),
                             1 - numpy.mean(err_rr),
                             numpy.mean(FE_r),
                             1 - numpy.mean(err_r)])

    else:
        out_file = []
        for pc in percentages[1:]:
            fnames = glob('errfe*_' + str(pc) + '_' + str(c) +'.txt')
            hund = fnames[2]
            fnames.remove(hund)
            fnames.append(hund)
            thou = fnames[1]
            fnames.remove(thou)
            fnames.append(thou)
            FE_0n, err_0n = numpy.loadtxt(fnames[0])
            out_file.append([pc, 0, numpy.mean(FE_0n), 1 - numpy.mean(err_0n), numpy.mean(FE_0n), 1 - numpy.mean(err_0n)])
            for fname, n in zip(fnames[1:], [25,50,100,1000]):
                FE_rr, err_rr, FE_r, err_r = numpy.loadtxt(fname)
                out_file.append([pc,
                                 n,
                                 numpy.mean(FE_rr),
                                 1 - numpy.mean(err_rr),
                                 numpy.mean(FE_r),
                                 1 - numpy.mean(err_r)])
    if save:
        save_name = str('errfe_stats' + str(c) + 'c.csv')
        header = 'pc,n,FE rr,err rr,FE r,err r'
        numpy.savetxt(save_name, out_file, delimiter=',',
                      fmt='%i,%i,%1.2f,%1.2f,%1.2f,%1.2f',
                      header=header, comments='')
    else:
        print 'FE_rr', 'err_rr', 'FE_r', 'err_r'
        print out_file
        
def stats_c(save=False):
    out_file_rr = [] ; out_file_r = []
    for c in lengths:
        c_row_rr = [c] ; c_row_r = [c]
        for pc in percentages[1:]:
            fnames = glob('*' + str(pc) + '_' + str(c) + '.txt')
            hund = fnames[2]
            fnames.remove(hund)
            fnames.append(hund)
            thou = fnames[1]
            fnames.remove(thou)
            fnames.append(thou)
            #err_0n = numpy.loadtxt(fnames[0])[1]
            #c_row_rr.append(1 - numpy.mean(err_0n))
            #c_row_r.append(1 - numpy.mean(err_0n))
            for fname in fnames[1:]:
                c_row_rr.append(1 - numpy.mean(numpy.loadtxt(fname)[1]))
                c_row_r.append(1 - numpy.mean(numpy.loadtxt(fname)[3]))
        out_file_rr.append(c_row_rr)
        out_file_r.append(c_row_r)

    if save:
        header = 'c'
        for pc in range(len(percentages[1:])):
            for n in ns[1:]:
                header += ',' + str(n)

        save_name_rr = str('err_stats_c_rr.csv')
        fmt = ''
        for i in range(len(out_file_rr[0])):
            if i == 0:
                fmt += '%i'
            else:
                fmt += ',%1.2f'
        numpy.savetxt(save_name_rr, out_file_rr, delimiter=',', fmt=fmt,
                      header=header, comments='')
        save_name_r = str('err_stats_c_r.csv')
        numpy.savetxt(save_name_r, out_file_r, delimiter=',', fmt=fmt,
                      header=header, comments='')

    else:
        print out_file_rr
        print out_file_r



def energyBP(n, c, save=False):
    '''Creates 3 FE boxplots (control, recon., reclass.)'''
    fnames_0 = glob('errfe0*1.txt') ; fnames_0 = move_ten(fnames_0)
    control_FEs = []
    for control_set in percentages:
        control_FEs.append(numpy.loadtxt(fnames_0[control_set])[0])
    
    impz_rr = [] ; impz_r = []
    fnames = glob('errfe' + str(n) + '_*_' + str(c) + '.txt')
    if c == 1:
        fnames = move_ten(fnames)
    else:
        ten = fnames[0]
        fnames.remove(ten)
        fnames.append(ten)
        impz_rr = [control_FEs[0]] ; impz_r = [control_FEs[0]]
    for fname in fnames:
        impz_rr.append(numpy.loadtxt(fname)[0])
        impz_r.append(numpy.loadtxt(fname)[2])
    
    FE_max = numpy.round(numpy.amax(control_FEs[0]), 2)
    
    fig, (ax, ax_rr, ax_r) = plt.subplots(1, 3, figsize=(12,4), sharey=True)
    plt.subplots_adjust(wspace=0.1)
    ax.boxplot(control_FEs)
    ax.set_title('no reconstruction')
    ax.set_xlabel('% corruption')
    ax.set_ylabel('free energy')
    ax.set_ylim(-600,300)
    ax.xaxis.set_ticklabels(percentages)
    ax.text(0.7, -200, str(FE_max))
    for pc_set in control_FEs:
        ax.axhline(y=numpy.median(pc_set), c='0.25', linestyle=':')
    ax_rr.boxplot(impz_rr)
    ax_rr.set_title('reconstruction')
    ax_rr.set_xlabel('% corruption')
    ax_rr.xaxis.set_ticklabels(percentages)
    for pc_set in impz_rr:
        ax_rr.axhline(y=numpy.median(pc_set), c='0.25', linestyle=':')
    ax_rr.set_ylim(-600,300)
    ax_r.boxplot(impz_r)
    ax_r.set_title('reclassification')
    ax_r.set_xlabel('% corruption')
    ax_r.xaxis.set_ticklabels(percentages)
    ax_r.yaxis.tick_right()
    for pc_set in impz_r:
        ax_r.axhline(y=numpy.median(pc_set), c='0.25', linestyle=':')
    ax_r.set_ylim(-600,300)
    
    if save:
        save_name = str('FE_boxplots' + str(n) + 'n' + str(c) + 'c')
        fig.savefig(save_name, bbox_inches='tight')
    else:
        plt.show()

def errorLG(n, c, save=False):
    '''Creates a 3-line graph with the calculated errors.'''
    fnames_0 = glob('errfe0*1.txt') ; fnames_0 = move_ten(fnames_0)
    control_errs = []
    for control_set in percentages:
        control_errs.append(1 - numpy.mean(numpy.loadtxt(fnames_0[control_set])[1]))
        
    impz_rr = [] ; impz_r = []
    fnames = glob(str('errfe' + str(n) + '_*_' + str(c) + '.txt'))
    if c == 1:
        fnames = move_ten(fnames)
    else:
        ten = fnames[0]
        fnames.remove(ten)
        fnames.append(ten)
        impz_rr = [control_errs[0]] ; impz_r = [control_errs[0]]
    for fname in fnames:
        impz_rr.append(1 - numpy.mean(numpy.loadtxt(fname)[1]))
        impz_r.append(1 - numpy.mean(numpy.loadtxt(fname)[3]))

    
    fig = plt.figure(figsize=(5,3))
    plt.plot(percentages, control_errs, 'b-', label='No reclassification')
    plt.plot(percentages, control_errs, 'bo')
    plt.plot(percentages, impz_rr, 'g--', label='Reclass + Reconstr.')
    plt.plot(percentages, impz_rr, 'go')
    plt.plot(percentages, impz_r, 'r:', label='Reclassification')
    plt.plot(percentages, impz_r, 'ro')
    plt.title('Classification Error')
    plt.xticks(range(0, 11))
    plt.xlabel('% corruption')
    plt.ylabel('error rate')
    plt.legend(loc=0)

    if save:
        save_name = str('err_linegraph' + str(n) + 'n' + str(c) + 'c')
        fig.savefig(save_name, bbox_inches='tight')
    else:
        plt.show()

def fe_c(save=False):
    control_FE = []
    for pc_set in percentages[1:]:
        control_FEs = []
        fnames_0 = glob('errfe0_' + str(pc_set) + '_*.txt')
        fnames_0 = move_ten(fnames_0)
        for fname in fnames_0:
            control_FEs.append(numpy.loadtxt(fname)[0])
        control_FE.append(control_FEs)
        
    _25_FE = []
    for pc_set in percentages[1:]:
        _25_FEs = []
        fnames_0 = glob(str('errfe25_' + str(pc_set) + '_*.txt'))
        fnames_0 = move_ten(fnames_0)
        for fname in fnames_0:
            _25_FEs.append(numpy.loadtxt(fname)[0])
        _25_FE.append(control_FEs)

    
    fig, ((ax1, ax2, ax5, ax10),
          (ax251, ax252, ax255, ax2510)) = plt.subplots(2,4, figsize=(16,8),sharey=True)
    ax1.boxplot(control_FE[0])
    ax1.set_ylabel('FE at n=0')
    ax1.set_xlabel('length of corruption')
    ax2.boxplot(control_FE[1])
    ax5.boxplot(control_FE[2])
    ax10.boxplot(control_FE[3])
    ax251.boxplot(_25_FE[0])
    ax251.set_ylabel('FE at n=25')
    ax252.boxplot(_25_FE[1])
    ax255.boxplot(_25_FE[2])
    ax2510.boxplot(_25_FE[3])
    
    
    plt.show()


def img_save(digit_vector, savename):
    plt.imshow(-digit_vector.reshape(28,28), cmap=cm.gray)
    plt.xticks([])
    plt.yticks([])
    os.chdir(os.getcwd() + "/test_results")
    plt.savefig(savename)
    os.chdir('../')


def img(digit_vector):
    plt.imshow(-digit_vector.reshape(28,28), cmap=cm.gray)


'''
def pc4x4(pc_set):
    fnames = glob(str('errfe*_' + str(pc_set) + '*1.txt'))
    hund = fnames[2]
    fnames.remove(hund)
    fnames.append(hund)
    thou = fnames[1]
    fnames.remove(thou)
    fnames.append(thou)
    FE_0, err_0 = numpy.loadtxt(fnames[0])
    FEs_rr = [FE_0] ; FEs_r = [FE_0]
    errs_rr = [1 - numpy.mean(err_0)] ; errs_r = [1 - numpy.mean(err_0)]
    for fname in fnames[1:]:
        FE_rr, err_rr, FE_r, err_r = numpy.loadtxt(fname)
        FEs_rr.append(FE_rr)
        FEs_r.append(FE_r)
        errs_rr.append(1 - numpy.mean(err_rr))
        errs_r.append(1 - numpy.mean(err_r))
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, sharex=True, figsize=(8, 8))
    labels = [0,25,50,100,1000]
    ax0.set_title('FEs_Recon')    
    ax0.boxplot(FEs_rr)
    ax0.set_ylabel('FE')
    ax1.set_title('FEs_Reclass')
    ax1.boxplot(FEs_r)
    ax2.set_title("err_Recon")
    ax2.plot(errs_rr)
    ax2.set_ylabel('error rate')
    plt.xticks(range(1, 7), labels)
    ax3.set_title('err_Reclass')
    ax3.plot(errs_r)
    plt.xticks(range(1, 7), labels)
    plt.tight_layout
'''
# FE boxplots for n=0 and FE and err table
nrgserrs_0n(save=True)

# stat tables recon vs reclass for all c
# c = 0 for no corruption stats
for c in [0, 1, 2, 5, 10]:
    stats_pc(c, save=True)

# compare FE for no recon, recon, and reclass for all n and c
for n in [25, 50, 100, 1000]:
    for c in [1, 2, 5, 10]:
        energyBP(n, c, save=True)
# compare err for no recon, recon, and reclass for all n and c
for n in [25, 50, 100, 1000]:
    for c in [1, 2, 5, 10]:
        errorLG(n, c, save=True)

# generate images
# here five is saved as numpy.array(test_x.get_value(borrow=True)[165])
img_data = numpy.loadtxt('img_data.txt')
five, corr_five, corr_five_c5, nw, gen_five, gw, mix = img_data

