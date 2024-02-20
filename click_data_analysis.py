#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:14:36 2022

@author: tshan
"""

import numpy as np
import scipy.signal as signal
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
from expyfun.io import read_wav, write_hdf5
import mne

# %% IIR Filter function

def butter_highpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=1):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

# %% Parameters
# EEG param
eeg_fs = 10000
# Stim param
stim_fs = 48000
rate = 40
t_click = 60
n_epoch_click = 10
# ABR param
t_start, t_stop = -200e-3, 600e-3  # define ABR lag range
# %% Subject
subject_list = ['subject001', 'subject002', 'subject003','subject004' ,
                'subject005', 'subject006', 'subject007', 'subject008',
                'subject009', 'subject010', 'subject011', 'subject012',
                'subject013', 'subject015', 'subject016', 'subject017',
                'subject018', 'subject019', 'subject020', 'subject022', 
                'subject023', 'subject024']
# %% Analysis
for subject in subject_list:
    print('Processing ' + subject + '...')
    # Change to your own EEG data path
    eeg_path = "C:/Users/Administrator/Documents/ClickABR/ClickABR/click_EEG_dataset/" + subject + '_click_eeg.fif'
    eeg_raw = mne.io.read_raw_fif(eeg_path, preload=True)
    # EEG Preprocessing
    print('Filtering raw EEG data...')
    # High-pass filter
    eeg_f_hp = 1.
    eeg_raw._data = butter_highpass_filter(eeg_raw._data, eeg_f_hp, eeg_fs)
    # Notch filter
    notch_freq = np.arange(60, 180, 540)
    notch_width = 5
    for nf in notch_freq:
        bn, an = signal.iirnotch(nf / (eeg_fs / 2.), float(nf) / notch_width)
        eeg_raw._data = signal.lfilter(bn, an, eeg_raw._data)
    #Epoching
    print('Epoching EEG click data...')
    events, event_dict = mne.events_from_annotations(eeg_raw)
    epochs_click = mne.Epochs(eeg_raw, events, tmin=0,
                              tmax=(t_click - 1/stim_fs + 1),
                              event_id=1, baseline=None,
                              preload=True, proj=False)
    epoch_click = epochs_click.get_data()
    # Get x_out
    len_eeg = int(eeg_fs * t_click)
    x_out = np.zeros((n_epoch_click, 2, len_eeg))
    for i in range(n_epoch_click):
        x_out[i, :, :] = epoch_click[i, :, 0:int(eeg_fs*t_click)]
    x_out = np.mean(x_out, axis=1) # take the average of the two channels
    # Change to your own click waveform path
    file_path = "/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicABR/diverse_dataset/music_abr_diverse_beh/present_files/click/"
    # Converting click waveform to event pulse train
    print('Converting click waveform to event pulse train...')
    x_in = np.zeros((n_epoch_click, int(t_click * eeg_fs)), dtype=float)
    for ei in range(n_epoch_click):
        stim, fs_stim = read_wav(file_path + 'click{0:03d}'.format(ei) + '.wav')
        stim_abs = np.abs(stim)
        click_times = [(np.where(np.diff(s) > 0)[0] + 1) / float(fs_stim) for s in stim_abs]
        click_inds = [(ct * eeg_fs).astype(int) for ct in click_times]
        x_in[ei, click_inds] = 1
    
    # Deriving ABR in frequency domain
    print('Deriving ABR ...')
    # Do fft
    x_in_fft = fft(x_in, axis=-1)
    x_out_fft = fft(x_out, axis=-1)
    # Get cross-correlation
    cc = np.real(ifft(x_out_fft * np.conj(x_in_fft)))
    # Get the averaged ABR across 10 trials
    abr = np.mean(cc, axis=0)
    abr_response = np.concatenate((abr[int(t_start*eeg_fs):], abr[0:int(t_stop*eeg_fs)])) / (rate*t_click)
    lags = np.arange(start=t_start*1000, stop=t_stop*1000, step=1e3/eeg_fs) # time vector in unit 'ms'
    # Plotting
    #plt.plot(lags, abr_response)
    # Saving Click Response
    print('Saving click response...')
    write_hdf5('/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/click_EEG_dataset/click_abr_reponse' + '/' + subject + '_crosscorr_click.hdf5',
                dict(click_abr_response=abr_response, lags=lags), overwrite=True)
    
# %% Plot
fig = plt.figure(dpi=180)
fig.set_size_inches(3.5, 3)
plt.plot(lags, abr_response*1e6, c='k', linewidth=1, label='Click')
plt.xlim(-10, 30)
plt.ylim(-0.9,0.73)
plt.ylabel('Potential (μV)')
plt.xlabel('Time (ms)')
plt.grid(alpha=0.5)
plt.vlines(0,-0.9,0.73, linestyles='solid', color='k', linewidth=1, alpha=0.5)
plt.hlines(0,-10,30, linestyles='solid', color='k', linewidth=1, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('click_fig.png', format='png')
