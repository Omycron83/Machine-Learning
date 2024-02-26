#Author: Damian Grunert
#Date: 25-02-2024
#Content: A modular implementation of the default transformer architecture 
#using the Pytorch - Dynamic Computational Graph for Per-Sequence-Training and Prediction

import torch
import torch.nn as nn
from typing import Type
import torch.optim as optim
import torch.utils.data as data
import math
import copy
from abc import ABC

#Torch-Implementation:
import torch_transformer

#"-------------------------------- Low-Level Transformer Methods ------------------------------------------------"
#Implements the layer-normalization-layer in the transformer architecture.
#Not a class as it isnt parameterized.
def layerNorm(x):
    std_mean = torch.std_mean(x, dim = 0)
    return (x - std_mean[1]) / (std_mean[0] + 0.0001)

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model):
        super.__init__()
        self.h = h
        self.d_k = d_model // h #Can be chosen otherwise
        #Output matrix processing the concatenated attention heads
        self.WO = nn.parameter(torch.rand(self.d_k * h, d_model))

        #Learnt, general parameters to yield Q, K and V from an Input for all Attention Heads combined
        self.WQ_Comb = nn.parameter(torch.rand(d_model, d_model))
        self.WK_Comb = nn.parameter(torch.rand(d_model, d_model))
        self.WV_Comb = nn.parameter(torch.rand(d_model, d_model))

        #List of weight matrices for each attention head
        self.WQ = [nn.parameter(torch.rand(d_model, self.d_k) for i in range(h))]
        self.WK = [nn.parameter(torch.rand(d_model, self.d_k) for i in range(h))]
        self.WV = [nn.parameter(torch.rand(d_model, self.d_k) for i in range(h))]
        
    def attention(self, Q, K, V):
        softmax = nn.Softmax(dim = 0)
        return softmax((Q @ K)*(1 / math.sqrt(self.d_k))) @ V

    def forward(self, XQ, XK, XV):
        #Encoder has the same, decoder different inputs for the attention evaluation (from which Q, K, V are constructed)
        heads = [self.attention(XQ @ self.WQ_comb @ self.WQ[i], XK @ self.WK_comb @ self.WK[i], XV @ self.WV_comb @ self.WV[i]) for i in range(self.h)]
        return torch.cat(heads, dim = 1) @ self.WO

class MaskedMultiHeadAttention(MultiHeadAttention):
    def attention(self, Q, K, V):
        softmax = nn.Softmax(dim = 0)
        mask = torch.tril(torch.ones(Q.shape[0], K.shape[1]))
        return (softmax((Q @ K))* mask *(1 / math.sqrt(self.d_k))) @ V

#Implementation of one single ANN layer as used in most transformers with an additional residual connection
class ANNLayer(nn.Module):
    def __init__(self, d_ff, d_model):
        super.__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        self.weights_1, self.bias_1 = nn.parameter(nn.init.kaiming_normal_(torch.empty(d_ff, d_model))), nn.parameter(nn.init.ones_(torch.empty(d_ff)))
        self.weights_2, self.bias_2 = nn.parameter(nn.init.kaiming_normal_(torch.empty(d_model, d_ff))), nn.parameter(nn.init.ones_(torch.empty(d_model)))
    
    def forward(self, X):
        return nn.ReLU(X @ self.weights_1 + self.bias_1) @ self.weights_2 + self.bias_2

#"-------------------------------- Higher-Level Transformer-Components ------------------------------------------------"
def postionalEncoding(X):
    #Creates a 'transposed vector' of the sequence positions from 0 to the amount of sequence members
    pos_vec = torch.arange(0, X.shape[0]).unsqueeze(1)
    div_vec = torch.float_pow(torch.tensor(10000), torch.arange(0, X.shape[1], 2) / X.shape[1] * (-1))
    pos_enc = torch.empty(X.shape[0], X.shape[1])
    pos_enc[:, 0::2] = torch.sin(pos_vec * div_vec)
    pos_enc[:, 1::2] = torch.cos(pos_vec * div_vec)
    return X + pos_enc

class EmbeddingLayer(nn.Module, ABC):
    @ABC.abstractmethod
    def __init__(self, d_input, d_model):
        super.__init__()
        self.d_input = d_input
        self.d_model = d_model
    @ABC.abstractmethod
    def forward(self, x):
        pass

class EncoderLayer(nn.Module):
    def __init__(self, dim_ann, n_head, d_model):
        super.__init__()
        self.dim_ann = dim_ann
        self.n_head = n_head
        self.d_model = d_model
        self.attention = MultiHeadAttention(n_head, d_model)
        self.ann = ANNLayer(dim_ann, d_model)

    #We are assuming a post-LN-Architecture
    def forward(self, x):
        attended_val = self.attention(x, x, x)
        attended_res = attended_val + x
        attended_norm = layerNorm(attended_res)

        ann_val = self.ann(attended_norm)
        ann_res = ann_val + attended_norm
        ann_norm = layerNorm(ann_res)
        return ann_norm

class DecoderLayer(nn.Module):
    def __init__(self, dim_ann, n_head, d_model, has_encoder):
        super.__init__()
        self.dim_ann = dim_ann
        self.n_head = n_head
        self.d_model = d_model
        self.has_encoder = has_encoder
        self.attention = MultiHeadAttention(n_head, d_model)
        if has_encoder:
            self.enc_attention = MultiHeadAttention(n_head, d_model)
        self.ann = ANNLayer(dim_ann, d_model)

    def forward(self, x, enc_val = None):
        attended_val = self.attention(x)
        attended_res = attended_val + x
        attended_norm = layerNorm(attended_res)
        
        if self.has_encoder:
            attended_val_enc = self.attention(enc_val, enc_val, attended_norm)
            attended_res_enc = attended_val_enc + attended_norm
            attended_norm = layerNorm(attended_res_enc)
         
        ann_val = self.ann(attended_norm)
        ann_res = ann_val + attended_norm
        ann_norm = layerNorm(ann_res)
        return ann_norm

class OutputLayer(nn.Module, ABC):
    @ABC.abstractmethod
    def __init__(self, d_model, d_output):
        super.__init__()
        self.d_model = d_model
        self.d_output = d_output
        pass
    @ABC.abstractmethod
    def forward(self, x):
        pass

#"-------------------------------- Finished Implementations ------------------------------------------------"

#Complete transformer-architecture, entailing a variable number of encoder and decoder layers
class Transformer(nn.Module):
    def __init__(self, d_input: int, d_output: int, d_model: int, n_head: int, dim_ann: int, 
                 embedding_layer: Type[EmbeddingLayer], enc_layers: int, dec_layers: int, output_layer: Type[OutputLayer]):
        super.__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.dim_ann = dim_ann
        self.d_input = d_input
        self.d_output = d_output

        self.embedding_layer = embedding_layer(self.d_model)
        if enc_layers > 0:
            self.encoder = [EncoderLayer(dim_ann, n_head, d_model) for i in range(enc_layers)]
        if dec_layers > 0:
            self.decoder = [DecoderLayer(dim_ann, n_head, d_model, enc_layers > 0) for i in range(dec_layers)]
        self.output_layer = output_layer

    def forward(self, X, Output):
        #Embedding-Layer
        curr_repr = self.embedding_layer(X)
        assert int(curr_repr.shape[0]) == self.d_model
        #Positional Encoding Layer
        curr_repr = postionalEncoding(curr_repr)
        #Encoder-Block
        for i in range(self.enc_layers):
            curr_repr = self.encoder[i](curr_repr)
        #Decoder-Block
        for i in range(self.dec_layers):
            curr_repr = self.decoder[i](curr_repr, Output)
        #Output-Function
        return self.output_layer(curr_repr)

#"-------------------------------- Implementation Of Common High-Level Components ------------------------------------------------"



#"------------------------------------------------ Optimization Algorithms ------------------------------------------------"

#"-------------------------------- Unit-Tests ------------------------------------------------"
def test_pos_encoding():
    pass

def test_embedding():
    pass

def test_ann():
    pass

def test_multihead_attention():
    pass

def test_output_layer():
    pass

def test_encoder():
    pass

def test_decoder():
    pass