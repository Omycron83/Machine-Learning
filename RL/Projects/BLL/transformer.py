import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

#Torch-Implementation:
import torch_transformer
´

#"-------------------------------- Low-Level Transformer Methods ------------------------------------------------"
#Implements the layer-normalization-layer in the transformer architecture.
#Not a class as it isnt parameterized.
def layerNorm(x):
    std_mean = torch.std_mean(x, dim = 0)
    return (x - std_mean[1]) / (std_mean[0] + 0.0001)

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, is_encoder):
        super.__init__()
        self.is_encoder = is_encoder
        self.h = h
        self.d_k = d_model // h #Can be chosen otherwise
        #Output matrix processing the concatenated attention heads
        self.WO = nn.parameter(torch.rand(self.d_k * h, d_model))
        #If used in the encoder, one can take in the 
        if is_encoder:
            self.dim_0 = d_model
        else:
            self.dim_0 = self.d_k

        #List of weight matrices for each attention head
        self.WQ = [nn.parameter(torch.rand(self.dim_0, self.d_k) for i in range(h))]
        self.WK = [nn.parameter(torch.rand(self.dim_0, self.d_k) for i in range(h))]
        self.WV = [nn.parameter(torch.rand(self.dim_0, self.d_k) for i in range(h))]
        
    def attention(self, Q, K, V):
        softmax = nn.Softmax(dim = 0)
        return softmax((Q @ K)*(1 / math.sqrt(self.d_k))) @ V

    def forward(self, X, Q = None, K = None, V = None):
        if self.is_encoder:
            Q, K, V = X, X, X
        heads = [self.attention(Q @ self.WQ[i], K @ self.WK[i], V @ self.WV[i]) for i in range(self.h)]

        return torch.cat(heads, dim = 1) @ self.WO

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

#"-------------------------------- Transformer-Components ------------------------------------------------"
def postionalEncoding(X):
    #Creates a 'transposed vector' of the sequence positions from 0 to the amount of sequence members
    pos_vec = torch.arange(0, X.shape[0]).unsqueeze(1)
    div_vec = torch.float_pow(torch.tensor(10000), torch.arange(0, X.shape[1], 2) / X.shape[1] * (-1))
    pos_enc = torch.empty(X.shape[0], X.shape[1])
    pos_enc[:, 0::2] = torch.sin(pos_vec * div_vec)
    pos_enc[:, 1::2] = torch.cos(pos_vec * div_vec)
    return X + pos_enc

class EmbeddingLayer(nn.Module):
    def __init__(self, d_input, d_model):
        self.d_input = d_input
        self.d_model = d_model
        self.weight = nn.Parameter(nn.init.kaiming_normal_(torch.empty(d_model, d_input)))
    def forward(self, x):
        return x @ self.weight

class EncoderLayer(nn.Module):
    def __init__(self, dim_ann, n_head, d_model):
        super.__init__()
        self.dim_ann = dim_ann
        self.n_head = n_head
        self.d_model = d_model
        self.attention = MultiHeadAttention(n_head, d_model, True)
        self.ann = ANNLayer(dim_ann, d_model)

    #We are assuming a pre-LN-Architecture
    def forward(self, x):
        norm_input = layerNorm(x)
        attended_val = self.attention(norm_input)
        attended_res = attended_val + x

        attended_norm = layerNorm(attended_res)
        ann_val = self.ann(attended_norm)
        ann_res = ann_val + attended_res
        return ann_res

class DecoderLayer(nn.Module):
    def __init__(self, dim_ann, n_head, d_model):
        super.__init__()
        self.dim_ann = dim_ann
        self.n_head = n_head
        self.d_model = d_model
        self.attention = MultiHeadAttention(n_head, d_model, True)
        self.ann = ANNLayer(dim_ann, d_model)

    def forward(self, x, enc_val = None):
        pass

class OutputLayer(nn.Module):
    def __init__(self, d_model, d_output):
        super.__init__()
        pass
    def forward(self, x):
        pass

#"-------------------------------- Finished Implementations ------------------------------------------------"

#Complete transformer-architecture, entailing a variable number of encoder and decoder layers
class Transformer(nn.Module):
    def __init__(self, d_input, d_output, d_model, n_head, dim_ann, embedding_layer, enc_layers, dec_layers, output_layer):
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
            self.decoder = [DecoderLayer(dim_ann, n_head, d_model) for i in range(dec_layers)]
        self.output_layer = output_layer

    def forward(self, X, Output):
        Curr_Repr = self.embedding_layer(X)
        assert int(Curr_Repr.shape[0]) == self.d_model
        Curr_Repr = postionalEncoding(X)
        for i in range(self.enc_layers):
            Curr_Repr = self.encoder(Curr_Repr)
        for i in range(self.dec_layers):
            Curr_Repr = self.decoder(Curr_Repr, Output)

        return self.output_layer(Curr_Repr)


#"-------------------------------- Unit-Tests ------------------------------------------------"
