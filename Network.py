#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:10:13 2023

@author: tesstangney
"""


'''
Causal, Exponentially Dilated Variational Auto Encoder, 
adapted from Franceschi et al. 2019
https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/tree/master
'''


# MIT License
# 
# Copyright (c) [2023] [Tess Tangney]
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Encoder classes adapted from Franceschi et al. 2019
# https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/tree/master
# Created with the following license 

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


# Implementation of causal CNNs partly taken and modified from
# https://github.com/locuslab/TCN/blob/master/TCN/tcn.py, originally created
# with the following license.

# MIT License

# Copyright (c) 2018 CMU Locus Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.





import numpy
import torch
import pandas as pd
import matplotlib as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%

class Chomp1d(torch.nn.Module):
    
    """
    Removes the last elements of a time series.
    
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.
    
    @param chomp_size Number of elements to remove.
    """
    
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class SqueezeChannels(torch.nn.Module):
    
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """
    
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)


class CausalConvolutionBlock(torch.nn.Module):
    
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).
    
    @param in_channels Number of input channels.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param dilation Dilation parameter of non-residual convolutions.
    @param final Disables, if True, the last activation function.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False):
        super(CausalConvolutionBlock, self).__init__()

        # Computes left padding so that the applied convolutions are causal
        padding = (kernel_size - 1) * dilation

        # First causal convolution
        conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))

        # The truncation makes the convolution causal
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU()

        # Second causal convolution
        conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()

        # Causal network
        self.causal = torch.nn.Sequential(
            conv1, chomp1, relu1, conv2, chomp2, relu2
        )
        # Residual connection
        '''They are used to allow gradients to flow through a network 
        directly, without passing through non-linear activation functions. '''
        self.upordownsample = torch.nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

        # Final activation function
        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        print(numpy.shape(x))
        out_causal = self.causal(x)
        print('out_caual', numpy.shape(out_causal))
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)


class CausalCNN(torch.nn.Module):
    
    """
    Causal CNN, composed of a sequence of causal convolution blocks.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).
    
    @param in_channels Number of input channels.
    @param channels Number of channels processed in the network and of output
           channels.
    @param depth Depth of the network.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    
    def __init__(self, in_channels, channels, depth, out_channels,
                 kernel_size):
        super(CausalCNN, self).__init__()

        layers = []  # List of causal convolution blocks
        dilation_size = 1  # Initial dilation size

        for i in range(depth):
            in_channels_block = in_channels if i == 0 else channels
            layers += [CausalConvolutionBlock(
                in_channels_block, channels, kernel_size, dilation_size
            )]
            dilation_size *= 2  # Doubles the dilation size at each step

        # Last layer
        layers += [CausalConvolutionBlock(
            channels, out_channels, kernel_size, dilation_size, final = True
        )]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



class CausalCNNVariationalEncoder(torch.nn.Module):
    
    """
    Encoder of a time series using a causal CNN: the computed representation is
    the output of a fully connected layer applied to the output of an adaptive
    max pooling layer applied on top of the causal CNN, which reduces the
    length of the time series to a fixed size.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`).
    
    Addition: Reparametrised to make the latent space variational 
    
    @param in_channels Number of input channels.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    
    def __init__(self, in_channels, channels, depth, reduced_size,
                 out_channels, kernel_size):
        super(CausalCNNVariationalEncoder, self).__init__()
        causal_cnn = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels()  # Squeezes the third dimension (time)

        self.mu = torch.nn.Linear(reduced_size, out_channels)
        self.log_var = torch.nn.Linear(reduced_size, out_channels)
        self.kl = 0  # Keep track of KL Divergence
        
        self.network = torch.nn.Sequential(
            causal_cnn, reduce_size, squeeze
        )
        print("ENCODER", self.network)

    def forward(self, x):
        x = self.network(x)
        mu = self.mu(x)
        sigma = torch.exp(self.log_var(x))
        z = mu + sigma*torch.randn_like(sigma) # Reparametrisation
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()  # KL divergence  
        return z, mu
    
    
    
#%%

'''DECOCER'''

class DeChomp1d(torch.nn.Module):
    
    """
    Removes the last elements of a time series.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.
    
    @param chomp_size Number of elements to remove.
    """
    
    def __init__(self, chomp_size):
        super(DeChomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[: , : , self.chomp_size:]

class UnSqueezeChannels(torch.nn.Module):
    
    """
    Expandes to a thrid dimension
    """
    
    def __init__(self):
        super(UnSqueezeChannels, self).__init__()

    def forward(self, x):
        return x.unsqueeze(2)


class TransposeCausalConvolutionBlock(torch.nn.Module):
    
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).
    
    @param in_channels Number of input channels.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param dilation Dilation parameter of non-residual convolutions.
    @param final Disables, if True, the last activation function.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False):
        super(TransposeCausalConvolutionBlock, self).__init__()

        # Computes padding to be half the dilation
        padding = int(dilation/2)

        # First causal deconvolution
        self.deconv1 = torch.nn.utils.weight_norm(torch.nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=3,
            padding=padding, dilation=dilation
        ))

        # The truncation makes the deconvolution causal
        self.chomp1 = DeChomp1d(2*padding)
        self.relu1 = torch.nn.LeakyReLU()

        # Second causal deconvolution
        self.deconv2 = torch.nn.utils.weight_norm(torch.nn.ConvTranspose1d(
             out_channels, out_channels, kernel_size=3,
            padding=padding, dilation=dilation
        ))
        self.chomp2 = DeChomp1d(2*padding)

#         Residual connection
        '''They are used to allow gradients to flow through a network 
        directly, without passing through non-linear activation functions. '''
        self.upordownsample = torch.nn.ConvTranspose1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

#         Final activation function
        self.relu = None if final else torch.nn.LeakyReLU()

        self.causal = torch.nn.Sequential(self.deconv1, self.chomp1, self.relu1, self.deconv2, self.chomp2)
        
    def forward(self, x):

        out_causal = self.causal(x)

        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)


class CausalDeCNN(torch.nn.Module):
    
    """
    Causal DeCNN, composed of a sequence of causal deconvolution blocks.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).
    
    @param in_channels Number of input channels.
    @param channels Number of channels processed in the network and of output
           channels.
    @param depth Depth of the network.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    
    def __init__(self, in_channels, channels, depth, 
                 out_channels, kernel_size):
        super(CausalDeCNN, self).__init__()

        layers = []  # List of causal convolution blocks
        dilation_size = 2**(depth+1)  # Initial dilation size 

        for i in range(depth):
            in_channels_block = in_channels if i == 0 else channels
            layers += [TransposeCausalConvolutionBlock(
                in_channels_block, channels, kernel_size, dilation_size
            )]
            dilation_size = int(dilation_size/2)  # Halves the dilation size at each step

        # Last layer
        layers += [TransposeCausalConvolutionBlock(
            channels, 1, kernel_size, dilation_size, final=True
        )]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CausalCNNDecoder(torch.nn.Module):
    
    """
    De-encoder of a time series using a causal DeCNN: the computed representation is
    the output of a fully connected layer applied to the output of an adaptive
    max pooling layer applied on top of the causal CNN, which reduces the
    length of the time series to a fixed size.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`).
    
    @param in_channels Number of input channels.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    
    ADDITION data_length : The length of the timeseries data so that the
                        decoder knows what length of timesereis to recosntruct
    """
    
    def __init__(self, in_channels, channels, depth, reduced_size,
                 out_channels, kernel_size,data_length):
        super(CausalCNNDecoder, self).__init__()
        self.causal_cnn = CausalDeCNN(
            in_channels, channels, depth, 
            out_channels, kernel_size)
        # len of data
        self.increase_size = torch.nn.AdaptiveMaxPool1d(data_length)
        self.unsqueeze = UnSqueezeChannels()  # Adds 3rd dimension 
                             # linear_in, linear_out           
        self.linear = torch.nn.Linear(out_channels, in_channels) # fliped dimensions
        
        
        self.network = torch.nn.Sequential(self.linear, self.unsqueeze, 
                                           self.increase_size, self.causal_cnn)

    def forward(self, x):
             
        return self.network(x)


#%%

class Causal_VAE(torch.nn.Module):
    """Combins the Encoder and Decoder classes to form a variational 
    Auto-Encoder
    
    Returns
    -------
    self.decoder(z) : Tensor
                      Lightcurve generated by the decoder
                      
    mu : float
         The encoded mu value which was used to reparametrise 
    """
    
    def __init__(self, depth, latent_vars, data_length):
        """Constructor
        
        Parameters
        ----------
        nhidden : int
                  The number of nodes in the hidden layer
        latent_vars : int
                      The number of latent variables to encode the timeseries to
            """
    
        super(Causal_VAE, self).__init__()
        self.encoder = CausalCNNVariationalEncoder(in_channels = 1, channels=40, reduced_size=60, 
                           depth=depth, out_channels=latent_vars, kernel_size=3)
        
        self.decoder = CausalCNNDecoder(in_channels=60, channels=40, depth=depth, 
                                        reduced_size=1, out_channels=latent_vars,
                                        kernel_size=3, data_length=data_length)
        
    def forward(self, x):
        z, mu = self.encoder(x)
        return self.decoder(z), mu
#%%
''' Training '''

class Dataset(torch.utils.data.Dataset):
    """
    PyTorch wrapper for a numpy dataset.
    @param dataset Numpy array representing the dataset.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return numpy.shape(self.dataset)[0]

    def __getitem__(self, index):
        return self.dataset[index]
    

def validation_minibatch(model, test_data):
    
    """Calculates the loss function of the model 
    
    Shuffles the data and takes a batch of 32 to validate the network using
    the ELBO loss function
    
    Parameters
    ----------
    model : class
            Variational Autoencoder
    data : Tensor
           timeseries data of length 85 
    
    Returns
    -------
    int, the normalised loss of the network
    """
    
    test_torch_dataset = Dataset(test_data)
    test_generator = torch.utils.data.DataLoader(test_torch_dataset, 
                                                 batch_size=32, shuffle=True)

    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(test_generator):
            recon, enco = model(batch.to(device))

            loss = ((batch.to(device)-recon)**2).sum() + model.encoder.kl  # ELBO Loss function

            val_loss += loss.item()
    return val_loss/(i+1)


def Train(DATA, MODEL, epochs=200):
    
    """Trains the MODEL using minibatches
    
    Divids the data in to 80% training and 20% validation sets,
    Shuffles the training data and takes a batch of 32 to train the network
    using the ELBO loss function. Calls the validation_minibatch function and 
    passing the current model and testing set.
    
    
    Parameters
    ----------
    MODEL : class
            Variational Autoencoder
    DATA : Tensor
           The full dataset of timeseries of length 85 
    epochs : int
            default to 200, indicated the number of epochs to train for
    
    Returns
    -------
    MODEL : object
            The trained network
    loss_hist : list
                The loss of the training set for each epoch
    val_loss : list
                The loss of the validation set for each epoch
    """
    
    length = len(DATA) # To split the data 80:20, train:test
    train = DATA[:int(length*0.8)].astype(numpy.float32)
    test = DATA[int(length*0.8):].astype(numpy.float32)
    
    
    train_torch_dataset = Dataset(train)
    train_generator = torch.utils.data.DataLoader(train_torch_dataset, 
                                                  batch_size=32, shuffle=True)

    MODEL = MODEL.to(device)
    MODEL.train() 

    loss_hist = []
    val_loss = []
    optimizer = torch.optim.Adam(MODEL.parameters(), lr=1e-3) #.zero_grad()
    

    for epoch in range(epochs):
        print('Epoch: ', epoch)
        cum_loss = 0 
        for i, batch in enumerate(train_generator):
            optimizer.zero_grad()
            recon, enco = MODEL(batch.to(device))
            
            loss = ((batch.to(device)-recon)**2).sum() + MODEL.encoder.kl  # ELBO Loss function
            loss.backward()
            cum_loss +=loss.item()
            optimizer.step()

        print('Loss: ', loss.item())


        loss_hist.append(cum_loss/i) # Divide the cumulative loss by the number of batches
        val_loss.append(validation_minibatch(MODEL, test))

        
    return MODEL, loss_hist, val_loss


#%%

''' Example of how to train the network '''

filepath = "Path to dataset "

df = pd.read_csv(filepath, sep=" ", header=None)

# remove any headers or columns as needed then transform to numpy array
timeseries = df.to_numpy() 

# Normalising the Data with sklearn
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
timeseries_norm = scaler.fit_transform(timeseries)

# Expand to 3 dimensions
timeseries_3d = numpy.expand_dims(timeseries_norm[:,:],1) 

# Call the training function
trained_model , loss, validation_loss = Train(DATA = timeseries_3d,
                                         MODEL=Causal_VAE(depth=4, 
                                                latent_vars = 6,
                                                data_length=85),
                                                 epochs = 30)


