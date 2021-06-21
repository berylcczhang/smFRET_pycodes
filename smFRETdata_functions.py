#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 14:37:54 2021

@author: beryl C.Z.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def stacking(path):
    #path = './Toxin/210608/ch2_TcDA_SC_pH7_lp10_210608'
    data_file = [f for f in os.listdir(path) if f.endswith('good.dat')]
    data_stack = []
    
    for files in data_file:
        directory = path + '/' + files
        data = np.loadtxt(directory)
        data_stack.append(data)
    
    data_stack = np.concatenate(data_stack,axis=1)
    np.savetxt(path + '/stacked_data.dat', data_stack, fmt='%8.1f' )
    return data_stack

def pruning(data):
    x = np.linspace(11,data.shape[0],data.shape[0]-10)
    loop = int(data.shape[1]/2)
    pw_fret = []
    pruned_traces = []
    
    for i in range(loop):
        A = data[:,2*i+1]
        D = data[:,2*i]
        
        fig = plt.figure(figsize=(25,6))
        ax1 = fig.add_subplot(211)
        ax1.set_xlim(10,560)
        ax1.set_xlabel('number of frames')
        ax1.set_ylabel('D/A intensity')
        ax1.set_xticks(np.arange(10, x.shape[0]+1, 20));
        ax2 = fig.add_subplot(212)
        ax2.set_xlim(10,560)
        ax2.set_xlabel('number of frames')
        ax2.set_ylim(-0.1,1)
        ax2.set_ylabel('$E_{fret}$')
        ax2.set_xticks(np.arange(10, x.shape[0]+1, 20));
        fig.suptitle('Trace '+str(i+1)+'/'+str(loop), fontsize=10)
        plt.tight_layout()
        
        ax1.plot(x, A[10:],'r',linewidth=1)
        ax1.plot(x, D[10:],'b',linewidth=1)
        E_fret = A/(A+D)
        ax2.plot(x,E_fret[10:],'k', linewidth=1)
        plt.show()
        
        framenumber = int(input("At which frame the acceptor is bleached?"))
        pw_fret.append(E_fret[10:framenumber])
        pruned_traces.append(D[10:framenumber])
        pruned_traces.append(A[10:framenumber])

    pw_fret_all = np.concatenate(pw_fret)
    # np.savetxt('./Toxin/210608/ch2_TcDA_SC_pH7_lp10_210608/pwE_all.dat', pw_fret_all, fmt='%7.5f')
    traces_p = []
    for i in range(int(len(pruned_traces)/2)):
        frame_len = len(pruned_traces[2*i])
        intensity = np.array([np.ones(frame_len)*(i+1),pruned_traces[2*i],pruned_traces[2*i+1]])
        intensity = intensity.transpose()
        traces_p.append(intensity)
    traces_p = np.concatenate(traces_p)
    # traces_p.shape
    # np.savetxt('./Toxin/210608/ch2_TcDA_SC_pH7_lp10_210608/pruned_all.dat', traces_p, fmt='%8.1f')
    return pw_fret_all, traces_p

def plot_hist(data, titlename):
    plt.figure(figsize=(10,7))
    plt.hist(data, bins=100, weights=np.ones(len(data))/len(data) , color='#56B4E9', edgecolor='k');
    plt.xlabel('$E_{fret}$' ,  fontsize=18)
    plt.ylabel('Probability' , fontsize=18)
    plt.title(titlename)
    plt.show()