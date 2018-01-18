#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:26:26 2018

@author: KarimM
"""

import numpy as np 
from scipy.io import wavfile
import separate_ikala as voice_separation
import librosa


[fs, x] = wavfile.read('/Users/KarimM/GoogleDrive/PhD/Research/IoSLdatasetExtension/all/excerpt1_3OH3_Eyes_Closed_Omens.wav')

#%% VAR
def Calcullate_VAR(inputFileDir, InputFileName):
    #Perform Separation
    model_dir = "/Users/KarimM/GoogleDrive/PhD/Research/Tools/\
    DeepConvSep/Models/fft_1024.pkl"
    output_dir = "/Users/KarimM/GoogleDrive/PhD/Research/IoSLDataset/"
    inputFile = inputFileDir + InputFileName
    voice_separation.train_auto(inputFile,output_dir,model_dir)
    #Get energy ratio
    [fs, xVocals] = wavfile.read(output_dir + InputFileName[:-4] + "-voice" \
    + ".wav")
    [fs, xMusic] = wavfile.read(output_dir + InputFileName[:-4] + "-music" \
    + ".wav")
    vocalsEnergy = sum(abs(xVocals/65536.0)**2)
    musicEnergy = sum(abs(xMusic/65536.0)**2)
    VAR = vocalsEnergy/musicEnergy
    return VAR


#%% Librosa Baseline Features
def Calculate_baseline_features(inputFile):
    track, sr = librosa.load(inputFile)
    sc = librosa.feature.spectral_centroid(y=track, sr=sr)
    s_roll = librosa.feature.spectral_rolloff(y=track, sr=sr)
    zc = librosa.feature.zero_crossing_rate(y=track)
    mfcc = librosa.feature.mfcc(y=track, sr=sr, n_mfcc=5)
    #mfcc_delta = librosa.feature.delta(mfcc)
    rms = librosa.feature.rmse(y=track)
    onset_env = librosa.onset.onset_strength(track, sr=sr)
    tempo = float(librosa.beat.tempo(onset_envelope=onset_env, sr=sr))
    sc_mean = np.mean(sc)
    sc_dev = np.std(sc)
    s_roll_mean = np.mean(s_roll)
    s_roll_dev = np.std(s_roll)
    zc_mean = np.mean(zc)
    zc_dev = np.std(zc)
    mfcc_mean = np.mean(mfcc,axis =1)
    mfcc_dev = np.std(mfcc,axis =1)    
    #mfccdelta_mean = np.mean(mfcc_delta,axis =1)
    #mfccdetla_dev = np.std(mfcc_delta,axis =1)    
    rms_mean = np.mean(rms)
    rms_dev = np.std(rms)
    trackFeatures = []
    trackFeatures.extend(mfcc_mean)
    trackFeatures.extend(mfcc_dev)
    trackFeatures.extend([sc_mean,sc_dev,s_roll_mean,s_roll_dev,zc_mean,zc_dev
                          ,rms_mean,rms_dev,tempo])
    return trackFeatures