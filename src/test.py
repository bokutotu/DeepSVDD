import torch
from torchsummary import summary

from networks.BT import BT_Autoencoder

net = BT_Autoencoder().to("cuda")

summary(net,(3,128,128))
