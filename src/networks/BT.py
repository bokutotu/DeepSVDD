import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging


class Flatten(torch.nn.Module):
    """
    flatten class for nn.Sequencial
    """

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class ConvBlock(nn.Module):
    """
    ConvBlock for DeepSVDD

    """

    def __init__(self, in_channels, out_channels, kernel_size, padding,acti_func):
        super().__init__()
        self.net = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                      kernel_size= kernel_size, stride=1, padding=padding,bias=False),
                    nn.BatchNorm2d(out_channels,eps=1e-4),
                    getattr(nn, acti_func)(),
                    nn.MaxPool2d(2,2)
                )

    def forward(self, x):
        return self.net(x)


class Interpolate(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.interpolate(x,scale_factor=2)


class BTNet(nn.Module):
    """
    DeepSVDD networks
    Args:
        in_channels  : int  ==> input channels if color 3
        num_cunn     : int  ==> number of cnn layer
        channels     : list ==> cnn output channels list
        kernel_sizes : list ==> cnn kernel size list
        paddings     : list ==> cnn padding list
        rep_dim      : int  ==> output dimention
        linear       : int  ==> linear input demention
        acti_funcs   : list ==> name of cnn activate function
    """

    def __init__(self, in_channels, num_cnn, channels, kernel_sizes, paddings,
                 rep_dim, linear, acti_funcs):
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)

        self.net = nn.Sequential()

        self.rep_dim = rep_dim
        in_channels = in_channels
        for i in range(num_cnn):
            self.net.add_module("cnnblock{}".format(i),
                    ConvBlock(in_channels, out_channels=channels[i], kernel_size=kernel_sizes[i],
                              padding=paddings[i], acti_func=acti_funcs[i]))
            in_channels = channels[i]

        self.net.add_module("flatten", Flatten())
        self.net.add_module("linear",nn.Linear(linear,self.rep_dim))

    def forward(self, x):
        return self.net(x)

    def summary(self):
        """Network summary."""
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)


class BTAutoencoder(nn.Module):

    """
    deep svdd's autoencoder class for pretrain
    args:
        netcfg       : dict ==> BTNet config (encoder)
        channels     : list ==> decoder out channels
        kernel_sizes : list ==> decoder kernel_size of conv
        paddings     : list ==> decoder padding of conv
        strides      : list ==> decoder stride of conv
        acti_funcs   : list ==> decoder activation function
    """

    def __init__(self, netcfg, channels, kernel_sizes, paddings, strides, acti_funcs,interpolate):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.rep_dim = netcfg.rep_dim

        # encoder is same to BTNet
        self.encoder = BTNet(**netcfg)

        self.decoder = nn.Sequential()

        in_channels = int(netcfg.rep_dim / (4 * 4)) 
        for i in range(netcfg.num_cnn):
            self.decoder.add_module('convtranspose{}'.format(i),
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=channels[i],
                                       kernel_size=kernel_sizes[i], padding=paddings[i], 
                                       stride=strides[i], bias=False)),
            self.decoder.add_module('batchnorm{}'.format(i),nn.BatchNorm2d(channels[i])),
            self.decoder.add_module('actifunc{}'.format(i),getattr(nn, acti_funcs[i])())
            if interpolate[i]:
                self.decoder.add_module('interpolate{}'.format(i), Interpolate())
            in_channels = channels[i]

        self.decoder.add_module('output_conv',nn.Conv2d(in_channels=channels[-1],
                                                        out_channels=netcfg.in_channels,
                                                        kernel_size=3,padding=1,bias=False))
        self.decoder.add_module('output_func',getattr(nn,acti_funcs[-1])())    

    def forward(self,x):
        x = self.encoder(x)
        x = x.view(x.size(0), int(self.rep_dim / (4 * 4)), 4, 4)
        x = nn.LeakyReLU()(x)
        return self.decoder(x)

    def summary(self):
        """Network summary."""
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)


