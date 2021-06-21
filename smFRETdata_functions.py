#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 14:37:54 2021

@author: beryl
"""

import numpy as np
import matplotlib.pyplot as plt
import os

path = './Toxin/210608/ch2_TcDA_SC_pH7_lp10_210608'
data_file = [f for f in os.listdir(path) if f.endswith('good.dat')]
data_stack = []

for files in data_file:
    directory = path + '/' + files
    data = np.loadtxt(directory)
    data_stack.append(data)

data_stack = np.concatenate(data_stack,axis=1)

np.savetxt(path + '/stacked_data.dat', data_stack, fmt='%8.1f' )

