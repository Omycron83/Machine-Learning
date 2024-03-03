#Author: Damian Grunert
#Date: 25-02-2024
#Content: A modular implementation of an improved transformer architecture incorporating possible improvements, such as:
# - Vectorized and parallelizable mini-batch training and prediction
# - Pre-LN-Architecture
# - GeLU activation function
# - The DropOut-Mechanism for ANN and attention regularization
# - Removing the bias in feedforward-layers
# 
#

#List of weight matrices for each attention head
self.WQ = [nn.parameter(torch.rand(d_model, self.d_k) for i in range(h))]
self.WK = [nn.parameter(torch.rand(d_model, self.d_k) for i in range(h))]
self.WV = [nn.parameter(torch.rand(d_model, self.d_k) for i in range(h))]

def forward(self, X, XQ = None, XK = None, XV = None):
        if self.is_encoder:
            XQ, XK, XV = X, X, X #Encoder has the same, decoder different inputs for the attention matrices
        heads = [self.attention(XQ @ self.WQ[i], XK @ self.WK[i], XV @ self.WV[i]) for i in range(self.h)]

        return torch.cat(heads, dim = 1) @ self.WO


class EncoderLayer(nn.Module):
    def __init__(self, dim_ann, n_head, d_model):
        super.__init__()
        self.dim_ann = dim_ann
        self.n_head = n_head
        self.d_model = d_model
        self.attention = MultiHeadAttention(n_head, d_model, True)
        self.ann = ANNLayer(dim_ann, d_model)

    #We are assuming a post-LN-Architecture
    def forward(self, x):
        norm_input = layerNorm(x)
        attended_val = self.attention(norm_input)
        attended_res = attended_val + x

        attended_norm = layerNorm(attended_res)
        ann_val = self.ann(attended_norm)
        ann_res = ann_val + attended_res
        return ann_res
