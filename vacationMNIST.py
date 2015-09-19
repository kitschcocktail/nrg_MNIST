# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 11:11:24 2015

@author: heiligenstein
"""
import os
import numpy

from tenk import tenk

"""
This script runs the tenk.py file for n-iterations of reclassifications, 
per percentage and outage length. If interrupted, the loop positions are saved
in a file, and when rerun, picks up where it left off. 

The root directory is P:\Students\Kim\energy\MNIST
"""


def vacationMNIST():
    os.chdir("test_results")
    last_positions = numpy.loadtxt("positions.txt") # load positions where it left off
    last_n, last_pc, last_len = last_positions
    os.chdir("..")
    try:
        n_iters = numpy.array([25, 50, 100, 1000]) # done: 0
        n_index = numpy.where(n_iters == last_n)[0][0]
        n_iters = list(n_iters[n_index:])
        
        pc_corruption = numpy.array([1, 2, 5, 10])
        pc_index = numpy.where(pc_corruption == last_pc)[0][0]
        pc_corruption = list(pc_corruption[pc_index:])
        
        len_corruption = numpy.array([1, 2, 5, 10])
        len_index = numpy.where(len_corruption == last_len)[0][0]
        len_corruption = list(len_corruption[len_index:])

        for n in n_iters:
            for pc in pc_corruption:
                    for length in len_corruption:
                        print 'Executing tenk with parameters:', n, pc, length
                        tenk(n, pc, length)
                        positions = n, pc, length
                        os.chdir("test_results")
                        numpy.savetxt('positions.txt', positions, fmt='%1.0f')
                        os.chdir("..")
                    len_corruption = [1, 2, 5, 10]
            pc_corruption = [0, 1, 2, 5, 10]

        print 'Finished. Files saved in /test_results.'
        


    except KeyboardInterrupt:
        print 'Interrupted at:', n, pc, length
        positions = n, pc, length
        os.chdir("test_results")
        numpy.savetxt('positions.txt', positions, fmt='%1.0f')
        print "Last positions saved in file 'positions.txt'"
        os.chdir("..")
        
if __name__ == "__main__":
    vacationMNIST()
    