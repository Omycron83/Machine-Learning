import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

#Torch-Implementation:
import torch_transformer

#Implements the layer-normalization-layer in the transformer architecture
class LayerNorm(nn.Module):
    def __init__(self):
        pass
    def forward(self, x):
        pass

#
class MultiHeadAttention(nn.Module):
    def __init__(self):
        pass
    def forward(self, x):
        pass

class ANNLayer(nn.Module):
    def __init__(self):
        pass
    def forward(self, x):
        pass

class Encoder(nn.Module):
    def __init__(self, enc_layers, ANNLayer, MultiHeadAttention, LayerNorm):
        pass
    def forward(self, x):
        pass

class Decoder(nn.Module):
    def __init__(self, dec_layers, ANNLayer, MultiHeadAttention, LayerNorm):
        pass
    def forward(self, x):
        pass

class Transformer(nn.Module):
    def __init__(self, d_model, n_head, enc_layers, dec_layers, dim_ann, epsilon):
        if enc_layers > 0:
            self.encoder = Encoder()
        if dec_layers > 0:
            pass
    def forward(self, x):
        pass