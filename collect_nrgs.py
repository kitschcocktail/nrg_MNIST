# -*- coding: utf-8 -*-
"""
Created on Thu Aug 06 16:08:21 2015

@author: heiligenstein
"""

from kgenerate import generate_MNIST


no_imp = []
for i in range(100):
    no_imp.append(generate_MNIST(7, 10))
    
