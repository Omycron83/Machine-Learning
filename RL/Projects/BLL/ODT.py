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

#A replay buffer saving trajectories of the form (g_t, s_t, a_t) in three seperate lists, each containing tensors of len_traj x d_g/s/a
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
    def retrieve(self, K = None):
        if K == None:
            K = self.max_length
        traj_lengths = [g.shape[0] for g in self.g]
        chosen_traj_index = random.choice(list(range(len(self.g))), weights=traj_lengths) #Picks an index proportional to the length of the trajectory at that index

        if self.g[chosen_traj_index].shape[0] > K:
            slice_index = random.randint(0, self.g[chosen_traj_index].shape[0] - K)
            return self.g[slice_index:K + slice_index, :], self.s[slice_index:K + slice_index, :], self.a[slice_index:K + slice_index, :]
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
            
#------------------------------- Directly taken from Zheng et al, 2022 and the original Online Decision Transformer Paper -----------------
#Implements a transform to the normal distribution in order to 'squeeze' the output values to a range of [0, 1]
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
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - torch.nn.functional.F.softplus(-2.0 * x))
# ------------------------------------------------------------------------------------------------------------------------------------------

#Facilitates a distribution of the form batch_size x d_seq x d_action according a mean and variance for each entry being provided (i.e. tensors of batch_size x d_seq x d_action)
#There is nothing being 'learned' for these, however the projections to their mean and variance are
#We, on top of that normal distribution, then also apply a tan-h-transform in order to 
class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.base_dist = pyd.Normal(mean, std)
        self.transforms = [TanhTransform()]
        super().__init__(self.base_dist, self.transforms)

    #Just returns the transformed mean
    @property
    def mean(self):
        return self.transforms[0](self.mean)
        
    #Samples from the distribution, applies the empirical shannon entropy
    def entropy(self, N=1):
        x = self.rsample((N,))
        log_p = self.log_prob(x)
        return -log_p.mean(axis=0).sum(axis=2)

    #Returns the log-likelihood-estimate (the probablility of the logarithm of how likely an a action would have been) according to the current stochastic policy for each step in the trajectory
    def log_likelihood(self, x):
        return self.log_prob(x).sum(axis=2)

#Defines the error between an action distribution and the actual action taken, first part of our approximate dual problem evaluation step by not providing a gradient for the entropy_reg
def action_likelihood_loss(a_hat_dist, a, entropy_reg):
    #Average value of likelihood of current predictions
    log_likelihood = a_hat_dist.log_likelihood(a).mean()
    #Average value of entropy of current predictions
    entropy = a_hat_dist.entropy().mean()
    #We would like to maximize the total likelihood (i.e. minimize its additional inverse, thus choosing the negative-log-likelihood-loss) and apply a penalty for for the entropy with its value being 'fixed'
    loss = - log_likelihood - entropy_reg * entropy
    return loss, entropy

#Minimizes the difference between a target entropy and given entropy with a variable, gradient-returning entropy-regularization parameter lambda, second part of our approximate dual problem evaluation step by not providing a gradient for the actual entropy
def action_entropy_loss(entropy_reg, entropy, target_entropy):
    return entropy_reg * (entropy - target_entropy).detach()

#Computes the output distributions from the given representations by generating the means and standard deviations of each entry by linear transformations
class OutputDist(torch.nn.Module):
    def __init__(self, model_dim, dim_a, log_std_bounds = [-5.0, 2.0]) -> None:
        super().__init__()
        self.W_mu = torch.nn.Parameter(torch.rand(model_dim, dim_a))
        self.W_std = torch.nn.Parameter(torch.rand(model_dim, dim_a))
        self.log_std_bounds = log_std_bounds

    #Applying the corresponding linear transformations and furthermore applying the bounds of the log-std by a common soft-clamp
    def forward(self, x):
        mu, std = x @ self.W_mu, x @ self.W_std
        log_std = (self.log_std_bounds[0] + 0.5 * (self.log_std_bounds[1] - self.log_std_bounds[0]) * (std + 1)).exp()
        return SquashedNormal(mu, log_std)

#Outputs the suspected return given the state and an action taken, the 
class OutputLayer(torch.nn.Module):
    def __init__(self, model_dim, dim_s, dim_a, log_std_bounds = [-5.0, 2.0]) -> None:
        super.__init__()
        self.model_dim = model_dim

        #Stochastic policy
        self.OutputDist = OutputDist(model_dim, dim_a, log_std_bounds)

        #Deterministic, auxilliary return and state output layers
        self.W_s = torch.nn.Parameter(torch.rand(model_dim, dim_s))
        self.W_r = torch.nn.Parameter(torch.rand(model_dim, 1))

    def forward(self, curr_repr):
        curr_repr = curr_repr.reshape(curr_repr.shape[0], curr_repr.shape[1] // 3, 3, self.model_dim).permute(0, 2, 1, 3)
        #Reshapes the representation into a shape of batch_size x 3 [as when we first read-in the values, so that we get the arrays R, S, A, respectively] x seq_length x d_model
        R, S, A = curr_repr[:, 2] @ self.W_r, curr_repr[:, 2] @ self.W_s, self.OutputDist(curr_repr[:, 1])
        return R, S, A
    
#Embedding for the three inputs, used right after having split them off
class ODTEmb(torch.nn.Module):
    def __init__(self, d_state, d_action, d_model):
        self.WR = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(1, d_model))) # Embedding matrix for the rewards
        self.WS = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(d_state, d_model))) # Embedding matrix for the rewards
        self.WA = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(d_action, d_model))) # Embedding matrix for the rewards
        self.d_model = d_model

    #Takes in the padded lists of return, state and action tensors of the shape d_batch x d_max_sequence x d_r/s/a
    def forward(self, R, S, A):
        #Multiplying them with the encoding matrices so that they are all of size d_batch x d_max_sequence x d_model, and then concatenating them to a d_batch x d_max_sequence * 3 x d_model tensor
        R, S, A = R @ self.WR, S @ self.WS, A @ self.WA
        #Temporarily stacking the values along a new dimension, such that for each batch entry there are now three seperate sequences in the tensor
        curr_repr = torch.stack(R, S, A, dim = 1)
        #Swapping the dimension 1 (along which the three input-types are represented) and the dimension 2 (along which the datapoints are ordered) -> instead of having three different representations for d_seq datapoints, we have d_seq datapoints with the three entries 
        curr_repr.permute(0, 2, 1, 3)
        #Collapsing the d_seq datapoints with 3 values each to one tensor of the final desired size
        return curr_repr.reshape(R.shape[0], 3 * R.shape[1], self.d_model)

#In order to step the embedding layer application
class EmptyEmb(transformer_improved.EmbeddingLayer):
    def forward(self, x):
        return x
    
#In order to step the output layer application
class EmptyOutput(transformer_improved.OutputLayer):
    def forward(self, x):
        return x

#Inheriting from the improved transformer to properly process an input-sequence before passing it onto the regular model, assumes decoder-only architecture
class ODTTransformer(transformer_improved.Transformer):
    def __init__(self, d_state: int, d_action: int, d_model: int, n_head: int, dim_ann: int, dec_layers: int, max_seq_length: int):
        super().__init__(0, d_model, d_model, n_head, dim_ann, EmptyEmb, 0, dec_layers, EmptyOutput, max_seq_length)
        self.ODTEmb = ODTEmb(d_state, d_action, d_model)

    def forward(self, R, S, A, dropout=0):
        #Applying the ODT-specific embedding layer
        output_seq = ODTEmb(R, S, A)

        #One may notice that we applied the positional encoding to the 3-sequence, instead of to each R, S, A-sequence in seperate
        #I postulate that this helps differentiating the meaning of the three while still preserving a notion of distance across time
        #I will also for now not infuse additional information about the window-position in the entire trajectory, though I will experiment with that later on

        #No input X due to decoder-only (doesnt call it anyway), no adding of token as first token is always provided by the choosen return-to-go (no empty generation)
        curr_repr = super().forward(None, output_seq, dropout, add_token=False) 
        R_pred, S_pred, A_pred = OutputDist
        return R_pred, S_pred, A_pred

#High-level class implementing the ODT-algorithm using multiple other classes to enable rl training
class ODT:
    #As a GPT-Architecture is used, we 
    def __init__(self, d_state: int, d_action: int, d_model: int, n_head: int, dim_ann: int, dec_layers: int, context_length: int, gamma: float, env, action_range, temperature):
        #Initializes the function approximator in this type of task
        self.transformer = ODTTransformer(d_state, d_action, d_model, n_head, dim_ann, ODTEmb, 0, dec_layers, OutputDist, context_length)

        #Initializes the replay buffer to facilitate storing, retrieval and hindsight-return-labeling
        self.ReplayBuffer = ReplayBuffer(context_length, gamma)

        self.action_range = action_range
        self.context_length = context_length

        #The current online-trajectory being encountered, formed by three lists containing R, S and A respectively. This is a temporary value.
        self.current_traj = [[], [], []]

        #The environment acted on, has to implement:
        """
        env: performs a step in the environment; env.step(action) -> s_t+1, r_t, terminated
        reset: resets the environment data
        is_terminated: 
        """
        self.env = env

        #Used as the "second variable" in optimizing the entropy of the produced output distributions
        self.temperature = torch.tensor(temperature)
        self.temperature.requires_grad = True

    #Optimizing the model according to the replays currently found in the replay-buffer
    def train(self, iterations, batch_size, dropout, corr_optim, entr_optim, target_entropy = None):
        for i in range(iterations):
            #Retrieving data from replay buffer, make this be vectorized
            R, S, A = [], [], []
            for i in range(batch_size):
                r, s, a = self.ReplayBuffer.retrieve(self.context_length)
                R.append(r)
                S.append(s)
                A.append(a)
            
            #Padding the lists of torch arrays to have the right size, combining them to the respective arrays
            R, S, A = self.pad_inputs(R), self.pad_inputs(S), self.pad_inputs(A)
            
            #Getting the prediction for this part of the model
            R_pred, S_pred, A_pred = self.transformer.forward(R, S, A, dropout) 

            #Performing a gradient step for both losses sequentially, warm-up hopefully not necessary due to pre-LN architecture
            loss_corr, entropy = action_likelihood_loss(A_pred, A, self.temperature)
            loss_corr.backward()
            corr_optim(self.transformer.parameters()).step()

            loss_entr = action_entropy_loss(self.temperature, entropy, target_entropy)
            loss_entr.backward()
            entr_optim(self.transformer.parameters()).step()


    #Performing and logging a trajectory in the environment, such that a R, S, A with dimensions d_seq x 1/d_s/d_a are returned
    def env_step(self, target_rtg):
        #Appending the next return-to-go 

        #Getting the 
        R_pred, S_pred, A_pred = self.transformer.forward(self.current_traj[0], self.current_traj[1], self.current_traj[2])

        action = A_pred[-1]
        state, reward, terminated = self.env(action)
        self.current_traj[0].append(reward)
        self.current_traj[1].append(state)
        self.current_traj[2].append(action)

        if terminated:
            self.ReplayBuffer.add_online(self.current_traj[0], self.current_traj[1], self.current_traj[2])
            self.current_traj = [[], [], []]
            self.env.reset()
        
    #Running an iteration and logging the found reward, without returning to replay buffer
    def evaluate(self, env):
        pass

    def add_data(self, traj_list):
        self.ReplayBuffer.add_offline(traj_list)


def unit_test():
    actions = [torch.rand(50, 2) for i in range(100)]
    states = [torch.rand(50, 4) for i in range(100)]
    rewards = [torch.rand(50, 1) for i in range(100)]
    OnlineDecisionTransformer = ODT()