#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 18:18:28 2023

@author: tesstangney
"""

import matplotlib as plt
import numpy
import torch 


def Generate_FRB(VAE, SCALER, LV, num, save = False):
    '''
    Generates fake data using the decoder of a trained VAE, need to change the
    path of the saved images depending where you want them saved and also the 
    length of the noise to be added
    
    Parameters
    ------
    VAE : Objest
          The trained model
          
    SCALER : sklearn.preprocessing StandardScaler object
            The scaler that was used to normalise the training data
            
    LV : int
         The number of Latent Variables
    num : int
          The number of fake FRBs to generate
    save : Boolean
           if True save the plots to Fake_FRB folder
           
    Returns
    -------
    Fake_LC : list
              List of generated timeseries data
    '''
    VAE.eval()
    Fake_LC = []
    for i in range(num):
        draw = torch.randn(1, LV)
        with torch.no_grad():
            gen_3d = VAE.decoder(draw)
            gen_2d = gen_3d.squeeze(1).detach().numpy()
            FRB = SCALER.inverse_transform(gen_2d)
            FRB = FRB.squeeze()
            '''Add Gaussian noise'''
            noise = numpy.random.normal(0,1,127)
            frb_noise = FRB+noise
            
            Fake_LC.append(FRB)
        plt.figure()
        plt.ylabel('S/N')
        plt.xlabel('Time[ms]')
        plt.plot(frb_noise, label='With Gaussian noise', c='orange')

        plt.plot(FRB, c='black')
        plt.legend()
        if save == True:
            plt.savefig('Fake_FRB'+str(i),
                        dpi = 250)
            
    return Fake_LC