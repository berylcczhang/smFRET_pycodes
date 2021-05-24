#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 12:23:43 2021

@author: beryl
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, ifft

#plt.style.use('ggplot')
np.random.seed(20)

#-------------------------------------------------------------------------------
# Set up

# Time
t = np.linspace(0,1,1000)

# Frequencies in the signal
f1 = 20
f2 = 30

# Some random noise to add to the signal
noise = np.random.random_sample(len(t))

# Complete signal
y = 2*np.sin(2*np.pi*f1*t+0.2) + 3*np.cos(2*np.pi*f2*t+0.3) + noise*5

# The part of the signal we want to isolate
y1 = 2*np.sin(2*np.pi*f1*t+0.2)

# FFT of the signal
F = fft(y)

# Other specs
N = len(t)                              # number of samples
dt = 0.001                              # inter-sample time difference
w = np.fft.fftfreq(N, dt)               # list of frequencies for the FFT
pFrequency = np.where(w>=0)[0]          # we only positive frequencies
magnitudeF = abs(F[:len(pFrequency)])   # magnitude of F for the positive frequencies

#-------------------------------------------------------------------------------
# Some functions we will need

# Plots the FFT
def pltfft():
    plt.plot(pFrequency,magnitudeF)
    plt.xlabel('Hz')
    plt.ylabel('Magnitude')
    plt.title('FFT of the full signal')
    plt.grid(True)
    plt.show()

# Plots the full signal
def pltCompleteSignal():
    plt.plot(t,y,'b')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Full signal')
    plt.grid(True)
    plt.show()

# Filter function:
# blocks higher frequency than fmax, lower than fmin and returns the cleaned FT
def blockHigherFreq(FT,fmin,fmax,plot=False):
    for i in range(len(FT)):
        if (i>= fmax) or (i<=fmin):
            FT[i] = 0
    if plot:
        plt.plot(pFrequency,abs(FT[:len(pFrequency)]))
        plt.xlabel('Hz')
        plt.ylabel('Magnitude')
        plt.title('Cleaned FFT')
        plt.grid(True)
        plt.show()
    return FT

# Normalising function (gets the signal in a scale from 0 to 1)
def normalise(signal):
    M = max(signal)
    normalised = signal/M
    return normalised

#-------------------------------------------------------------------------------
# Processing

# Cleaning the FT by selecting only frequencies between 18 and 22
newFT = blockHigherFreq(F,18,22)

# Getting back the cleaned signal
cleanedSignal = ifft(F)

# Error
error = normalise(y1) - normalise(cleanedSignal)

#-------------------------------------------------------------------------------
# Plot the findings

pltCompleteSignal()         #Plot the full signal
pltfft()                    #Plot fft

plt.figure()

plt.subplot(3,1,1)          #Subplot 1
plt.title('Original signal')
plt.plot(t,y,'g')

plt.subplot(3,1,2)          #Subplot 2
plt.plot(t,normalise(cleanedSignal),label='Cleaned signal',color='b')
plt.plot(t,normalise(y1),label='Signal to find',ls='-',color='r')
plt.title('Cleaned signal and signal to find')
plt.legend()

plt.subplot(3,1,3)          #Subplot 3
plt.plot(t,error,color='r',label='error')
plt.show()