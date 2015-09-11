# -*- coding: utf-8 -*-
"""
Created on Mon Sep 07 09:53:02 2015

@author: heiligenstein
"""
from logistic_sgd import load_data    
from graphs import plots
        
dataset = 'mnist.pkl.gz'
datasets = load_data(dataset)

percentages = [0, 2, 5, 10]

plots(datasets[2], percentages)

