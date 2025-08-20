#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 12:54:38 2021

@author: tong
"""
import numpy as np
from expyfun import ExperimentController
from expyfun._utils import set_log_level
from expyfun.stimuli import write_wav, read_wav
from expyfun._trigger_controllers import decimals_to_binary
import os
import shutil
import scipy.signal as sig
import random
import matplotlib.pyplot as plt

######## Making click train ##############
click_path = '/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/Muspeech_Dataset_2020_1/spectral_match/click'

is_Poisson = True
rate = 40
fs = 48000
dur = 60
trial_len = int(np.round(dur * fs))
n_trial = 10
click_dur = 100e-6
click_len = int(np.round(click_dur * fs))
click_dur = click_len / fs
ref_rms =0.01
n_channel = 1

click = 2 * ref_rms * 2 ** 0.5 * np.ones(click_len)  # ppeSPL

def di(rate, fs, dur):
    return np.random.exponential(fs / rate, int(np.ceil(rate * dur))).astype(int)


def inds2train(inds, trial_len):
    inds = inds[inds <= trial_len]
    train = np.zeros(trial_len)
    for i in inds:
        coin = random.randint(1, 2)
        if coin == 1:
            train[i] = -1.
        elif coin == 2:
            train[i] = 1.
    return sig.lfilter(click, 1, train, axis=-1)

for ti in range(n_trial):
    if is_Poisson:
        inds = []
        for _ in range(n_channel):
            inds_temp = np.append([1], np.cumsum(di(rate, fs, dur) +
                                                 click_len + 1))
            while inds_temp[-1] < dur * fs:
                inds_temp = np.append(inds_temp, inds_temp[-1] + np.cumsum(
                                      di(rate, fs, float(
                                         trial_len - inds_temp[-1]) + 0)))
            inds += [inds_temp]
        train = np.array([inds2train(i, trial_len) for i in inds])
        write_wav(click_path + '/train%03i.wav' % ti, train, fs)
            
#plt.plot(np.arange(0,dur,step=1/fs),train[0,:])


