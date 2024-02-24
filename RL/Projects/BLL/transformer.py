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


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, is_encoder):
        super.__init__()
        self.is_encoder = is_encoder
        self.d_k = d_model // h #Can be chosen otherwise
        #Output matrix processing the concatenated attention heads
        self.WO = nn.parameter(torch.rand(self.d_k * h, d_model))
        #If used in the encoder, one can take in the 
        if is_encoder:
            self.dim_0 = self.d_model
        else:
            self.dim_0 = self.d_k

        #List of weight matrices for each attention head
        self.WQ = [nn.parameter(torch.rand(self.dim_0, self.d_k) for i in range(h))]
        self.WK = [nn.parameter(torch.rand(self.dim_0, self.d_k) for i in range(h))]
        self.WV = [nn.parameter(torch.rand(self.dim_0, self.d_k) for i in range(h))]
        

    def attention(self, Q, K, V):
        

    def forward(self, x, Q = None, K = None):
        if self.is_encoder:
            Q, K = x, x

        return torch.cat()

#
class ANNLayer(nn.Module):
    def __init__(self):
        pass
    def forward(self, x):
        pass

#
class Encoder(nn.Module):
    def __init__(self, enc_layers, ANNLayer, MultiHeadAttention, LayerNorm):
        pass
    def forward(self, x):
        pass

#
class Decoder(nn.Module):
    def __init__(self, dec_layers, ANNLayer, MultiHeadAttention, LayerNorm):
        pass
    def forward(self, x):
        pass

#Complete transformer-architecture, entailing a variable number of encoder and decoder lazyers
class Transformer(nn.Module):
    def __init__(self, d_model, n_head, enc_layers, dec_layers, dim_ann, epsilon):
        if enc_layers > 0:
            self.encoder = Encoder()
        if dec_layers > 0:
            pass
    def forward(self, x):
        pass