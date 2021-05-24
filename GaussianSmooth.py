#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:24:43 2021

@author: beryl
"""
#def edgedetect(data):

import numpy.random
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

#Gaussian kernel (not normalized here)
def gaussian(x, sigma):
    return np.exp(-(x/sigma)**2/2) 

#convolution
def smooth(y, box_pts, std_norm): 
    x = (np.linspace(-box_pts/2.,box_pts/2.,box_pts + 1)) #Gaussian centred on 0
    #3. is an arbitrary value for normalizing the sigma
    sigma = box_pts/std_norm 
    integral = quad(gaussian, x[0], x[-1], args=(sigma))[0]
    box = gaussian(x, sigma)/integral
    y_smooth = np.convolve(y,box,mode='same')
    
    plt.figure(figsize=(20,4))
    plt.plot(y)
    plt.plot(y_smooth)
    
def edgedetect(y,step):
    grad = []
    for i in range(len(y)):
        grad_i = (np.mean(y[i+1:i+1+step])-np.mean(y[i:i+step]))
        grad.append(grad_i)
    grad = np.array(grad)
    return grad

        

    
    
