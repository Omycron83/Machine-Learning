#Author: Damian Grunert
#Date: 25-02-2024
#Content: A custom implementation of the online decision transformer algorithm using pytorch

from typing import Type
import transformer_improved
import torch
from torch import distributions as pyd
import math
from copy import deepcopy
import random

#A replay buffer saving trajectories of the form (g_t, s_t, a_t) in three seperate lists, each tensors of len_traj x d_g/s/a
#This enables us to treat each of them as a seperate sequence for now, and then link them during the ODT implementation
#Where g_t is a scalar and s_t and a_t are torch tensors of 1 x d_s and 1 x d_a fixed, respectively
class ReplayBuffer:
    def __init__(self, max_length, gamma) -> None:
        self.max_length = max_length
        self.gamma = gamma
        self.g = []
        self.s = []
        self.a = []

    #Randomly samples from trajectories according to relative size, and then from length-K subtrajectories from the chosen trajectory
    def retrieve(self, K):
        traj_lengths = [g.shape[0] for g in self.g]
        chosen_traj_index = random.choice(list(range(len(self.g))), weights=traj_lengths) #Picks an index proportional to the length of the trajectory at that index

        if self.g[chosen_traj_index].shape[0] > K:
            slice_index = random.randint(0, self.g[chosen_traj_index].shape[0] - K)
            return self.g[slice_index:K + slice_index, :], self.s[slice_index:K + slice_index, :] ,self.a[slice_index:K + slice_index, :]
        else:
            return self.g[chosen_traj_index], self.s[chosen_traj_index], self.a[chosen_traj_index]
    

    def hindsight_return_labeling(self, R):
        #Making a deep-copy as to mitigate external errors in reuse trajectories
        traj_list = deepcopy(traj_list)
        #Updating trajectory return-to-go as discounted sum in hindsight return labeling in monte-carlo fashion
        for j in range(len(traj_list)):
            for i in range(1, traj_list[j].shape[0] + 1):
                if i != 1:
                    traj_list[j][-i, 0] += self.gamma * traj_list[j][-(i - 1), 0]
        return traj_list

    #Inputs a list of lists for r, s and a (e.g. R = [[1, 1, 1], [2, 3, 4], [5, 43, 5]], A = [[tensor_1, ...], [tensor_1, ...], [tensor_1, ...]])
    def add_offline(self, R, S, A):
        self.a.append(A)
        self.g.append(self.hindsight_return_labeling(R))
        self.s.append(S)

        #Sorting the list and deleting 'over-the-top' elements, done according to the first element of the r-Tensors 
        self.a, self.g, self.s = zip(*zip(self.a, self.g, self.s).sort(key = lambda triple_trajectories: float(triple_trajectories[1][0, :])))
        if len(self.content) > self.max_length:
            self.a[-(len(self.a) - self.max_length):] = []
            self.g[-(len(self.g) - self.max_length):] = []
            self.s[-(len(self.s) - self.max_length):] = []
        
    #Assumes input lists for R, S and A
    def add_online(self, R, S, A):
        self.a.append(A)
        self.g.append(self.hindsight_return_labeling(R))
        self.s.append(S)
        if len(self.a) > self.max_length:
            self.a.pop(0)
            self.g.pop(0)
            self.s.pop(0)

#Used in continuous action spaces, where we just predict an output sequence
            
#------------------------------- Directly taken from Zheng et al, 2022 and the original Online Decision Transformer Paper -----------------
class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))

class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    """
    Squashed Normal Distribution(s)

    If loc/std is of size (batch_size, sequence length, d),
    this returns batch_size * sequence length * d
    independent squashed univariate normal distributions.
    """

    def __init__(self, loc, std):
        self.loc = loc
        self.std = std
        self.base_dist = pyd.Normal(loc, std)

        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self, N=1):
        # sample from the distribution and then compute
        # the empirical entropy:
        x = self.rsample((N,))
        log_p = self.log_prob(x)

        # log_p: (batch_size, context_len, action_dim),
        return -log_p.mean(axis=0).sum(axis=2)

    def log_likelihood(self, x):
        # log_prob(x): (batch_size, context_len, action_dim)
        # sum up along the action dimensions
        # Return tensor shape: (batch_size, context_len)
        return self.log_prob(x).sum(axis=2)
# ------------------------------------------------------------------------------------------------------------------------------------------

class OutputDist(torch.nn.Module):
    def __init__(self, model_dim, dim_a, log_std_bounds = [-5.0, 2.0]) -> None:
        super().__init__()
        self.W_mu = torch.nn.Parameter(torch.rand(model_dim, dim_a))
        self.W_std = torch.nn.Parameter(torch.rand(model_dim, dim_a))
        self.log_std_bounds = log_std_bounds

    def forward(self, x):
        mu, std = x @ self.W_mu, x @ self.W_std
        log_std = (self.log_std_bounds[0] + 0.5 * (self.log_std_bounds[1] - self.log_std_bounds[0]) * (std + 1)).exp()
        return SquashedNormal(mu, log_std)

#Used in discrete action space - to do for later implementations for discrete action spaces
class OutputDistDisc(transformer_improved.OutputLayer):
    def forward(self, x):
        return 
    
#Embedding for the three inputs, used right after having split them off
class ODTEmb(torch.nn.Module):
    def __init__(self, d_state, d_action, d_model):
        self.WR = torch.nn.Parameter(torch.rand(1, d_model)) # Embedding matrix for the rewards
        self.WS = torch.nn.Parameter(torch.rand(d_state, d_model)) # Embedding matrix for the rewards
        self.WA = torch.nn.Parameter(torch.rand(d_action, d_model)) # Embedding matrix for the rewards

    def forward(self, R, S, A):
        #Concatenizing the padded lists of return, state and action tensors to lists of the shape d_batch x d_max_sequence x d_r/s/a
        R, S, A = torch.cat(R, dim = 0).view(len(R), R[0].shape[0], R[0].shape[1]), torch.cat(S, dim=0).view(len(S), S[0].shape[0], S[0].shape[1]), torch.cat(A, dim = 0).view(len(A), A[0].shape[0], A[0].shape[1])
        #Multiplying them with the encoding matrices so that they are all of size d_batch x d_max_sequence x d_model, and then concatenating them to a d_batch x d_max_sequence x d_model tensor
        return torch.cat(R @ self.WR, S @ self.WS, A @ self.WA, dim = 2) 

#There is no real mask for the representation formed
class EmptyEmb(transformer_improved):
    def forward(self, x):
        return x

#Inheriting from the improved transformer to properly process an input-sequence before passing it onto the regular model, assumes decoder-only architecture
class ODTTransformer(transformer_improved.Transformer):
    def __init__(self, d_output: int, d_model: int, n_head: int, dim_ann: int, dec_layers: int, max_seq_length: int, d_state: int, d_action: int):
        super().__init__(0, d_output, d_model, n_head, dim_ann, EmptyEmb, 0, dec_layers, OutputDist, max_seq_length)
        self.ODTEmb = ODTEmb(d_state, d_action, d_model)

    def forward(self, R, S, A, dropout=0):
        R, S, A = self.pad_inputs(R), self.pad_inputs(S), self.pad_inputs(A)
        output_seq = ODTEmb(R, S, A)
        curr_repr = super().forward(None, output_seq, dropout) #No input X due to decoder-only (doesnt call it anyway)
        #Now, we 


#High-level class implementing the ODT-algorithm using multiple other classes to enable rl training
class ODT:
    #As a GPT-Architecture is used, we 
    def __init__(self, d_input: int, d_output: int, d_model: int, n_head: int, dim_ann: int, dec_layers: int, context_length: int, gamma: float, lr: float, env):
        #Initializes the function approximator in this type of task
        self.transformer = transformer_improved.Transformer(d_input, d_output, d_model, n_head, dim_ann, ODTEmb, 0, dec_layers, OutputDist, context_length)

        self.ReplayBuffer = ReplayBuffer(context_length, gamma)

        #Warm-up not needed due to LN-setup, but might try it out later
        #Optimizer for the correctness
        self.AdamWCorr = torch.optim.AdamW(lr = lr)
        #Optimizer for the entropy in the distributionw
        self.AdamWEntr = torch.optim.AdamW(lr = lr)

        #The current online-trajectory being encountered, formed by three lists containing R, S and A respectively. This is a temporary value.
        self.current_traj = [[], [], []]

        #The environment acted on, has to implement:
        """
        env: performs a step in the environment; env.step(action) -> s_t+1, r_t, terminated
        reset: resets the environment data
        is_terminated: 
        """
        self.env = env


    #Optimizing the model according to the replays currently found in the replay-buffer
    def train(self, iterations, lr, batch_size):
        #Retrieving data from replay buffer, make this be vectorized
        replays = []
        for i in range(batch_size):
            replays.append(self.ReplayBuffer.retrieve())
        
        #Getting the prediction for this part of the model
        output = self.transformer.forward(replays)
        loss_corr = None
        loss_entr = None

        #Performing a gradient step for both losses sequentially
        loss_corr.backward()
        self.AdamWCorr(self.transformer.parameters()).step()
        loss_entr.backward()
        self.AdamWEntr(self.transformer.parameters()).step()


    #Performing and logging a step made in an online-finetuning-step
    def env_step(self, current_traj):
        action = self.transformer.forward(current_traj)
        state, reward, terminated = self.env(action)
        self.current_traj.append((reward, state, action))
        if terminated:
            self.ReplayBuffer.add_online(self.current_traj)
            self.current_traj = []

    def add_data(self, traj_list):
        self.ReplayBuffer.add_offline(traj_list)