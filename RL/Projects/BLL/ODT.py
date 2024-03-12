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
import pickle

#A replay buffer saving trajectories of the form (g_t, s_t, a_t) in three seperate lists, each containing lists of tensors of shape 1 x d_g/s/a
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
        traj_lengths = [len(g) for g in self.g]
        chosen_traj_index = random.choice(list(range(len(self.g))), weights=traj_lengths) #Picks an index proportional to the length of the trajectory at that index

        if len(self.g[chosen_traj_index]) > K:
            slice_index = random.randint(0, len(self.g[chosen_traj_index]) - K)
            return self.g[chosen_traj_index][slice_index:K + slice_index], self.s[chosen_traj_index][slice_index:K + slice_index], self.a[chosen_traj_index][slice_index:K + slice_index]
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

    #Apparently, they thought of the fact that:
    # log(1 - tanh(x)^2) = log(sech(x)^2) = 2*log(sech(x)) = 2 * log(2e^-x / (e^-2x + 1)) = 2* (log(2) - x - softplus(-2x)) did some jacobian shit
    def log_abs_det_jacobian(self, x, y):
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
        
    #Samples from the distribution (once per default), applies the empirical shannon entropy
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
        self.OutputLayer = OutputLayer(d_model, d_state, d_action)

    def forward(self, R, S, A, dropout=0, add_token = True):
        #Applying the ODT-specific embedding layer
        output_seq = ODTEmb(R, S, A)

        #One may notice that we applied the positional encoding to the 3-sequence, instead of to each R, S, A-sequence in seperate
        #I postulate that this helps differentiating the meaning of the three while still preserving a notion of distance across time
        #I will also for now not infuse additional information about the window-position in the entire trajectory, though I will experiment with that later on

        #No input X due to decoder-only (doesnt call it anyway), no adding of token as first token is always provided by the choosen return-to-go (no empty generation)
        curr_repr = super().forward(None, output_seq, dropout, add_token)
        R_pred, S_pred, A_pred = self.OutputLayer(curr_repr)
        return R_pred, S_pred, A_pred

#High-level class implementing the ODT-algorithm using multiple other classes to enable rl training
class ODT:
    #As a GPT-Architecture is used, we 
    def __init__(self, d_state: int, d_action: int, d_model: int, n_head: int, dim_ann: int, dec_layers: int, context_length: int, replay_buffer_length: int, gamma: float, action_range, init_temperature = 0.1):
        self.d_state = d_state
        self.d_action = d_action
        self.context_length = context_length
        self.action_range = action_range
        self.context_length = context_length

        #Initializes the function approximator in this type of task
        self.transformer = ODTTransformer(d_state, d_action, d_model, n_head, dim_ann, ODTEmb, 0, dec_layers, OutputDist, context_length)

        #Initializes the replay buffer to facilitate storing, retrieval and hindsight-return-labeling
        self.ReplayBuffer = ReplayBuffer(replay_buffer_length, gamma)

        #Used as the "second variable" in optimizing the entropy of the produced output distributions
        self.temperature = torch.tensor(init_temperature)
        self.temperature.requires_grad = True

    #Optimizing the model according to the replays currently found in the replay-buffer
    def train(self, iterations, batch_size, corr_optim, entr_optim, target_entropy = None, norm_clip = 0.25, dropout = 0):
        if target_entropy == None:
            target_entropy = - self.action_range
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
            #We also clip the gradient of this in order to 
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), norm_clip)
            corr_optim(self.transformer.parameters()).step()

            loss_entr = action_entropy_loss(self.temperature, entropy, target_entropy)
            loss_entr.backward()
            entr_optim(self.transformer.parameters()).step()


    #Performing and logging a trajectory for a number of same environments, such that a R, S, A are added as lists of length 1 x 1/d_s/d_a are added to the replay directory
    """
    env: performs a step in the environment; env.step(action) -> s_t+1, r_t, terminated
    reset(): resets the environment data
    """

    def sample_traj(self, target_rtg, envs):
        has_finished = [False for i in range(len(envs))]

        states = [[i.reset()] for i in envs]
        rgs = [[torch.tensor(target_rtg).reshape(1, 1)] for i in range(len(envs))]

        #Actions are initialized with 0 and then filled in latr tm
        actions = [[torch.zeros(1, self.d_action)] for i in range(len(envs))]
        #Rewards are logged seperately latr tm
        rewards = [[] for i in envs]

        episode_return = 0
        while True:
            #Makes them to 
            R, S, A = self.pad_inputs(rgs), self.pad_inputs(states), self.pad_inputs(actions)

            #Getting the prediction for the current rtg, state and action sequence for 0-action
            R_pred, S_pred, A_pred = self.transformer.forward(R, S, A, add_token=False)

            action = A_pred[1, -1, :].sample().reshape(len(envs), )
            action.clamp(*self.action_range)

            for env in envs:
                pass
                        
            #Appending the next return-to-go to the current return-to-go-save using g_t+1 = (g_t - r_t) / gamma (which makes sense, as g_t-1 = gamma * g_t + r_t-1 <=> g_t = (g_t-1 + r_t-1) / gamma
            

            episode_return += reward
            if terminated:
                break

        self.ReplayBuffer.add_online(self.current_traj[0], self.current_traj[1], self.current_traj[2])
        self.env.reset()
        return reward
        
    #Running an iteration and logging the found reward, without returning to replay buffer
    def evaluate(self, env):
        pass

    def add_data(self, R, S, A):
        self.ReplayBuffer.add_offline(R, S, A)


def unit_test():
    actions = [torch.rand(50, 2) for i in range(100)]
    states = [torch.rand(50, 4) for i in range(100)]
    rewards = [torch.rand(50, 1) for i in range(100)]
    class DebugEnv:
        def __init__(self) -> None:
            self.opt_action = torch.ones(1, 4)
        def reset(self):
            return torch.zeros(1, 4)
        def step(self, action):
            return torch.random(1, 4), - torch.cdist(self.opt_action, action).reshape(1, 1)
        
    OnlineDecisionTransformer = ODT(4, 2, 8, 1, 124, 2, 10, 5, 1, [-1000, 1000])
    OnlineDecisionTransformer.add_data(rewards, states, actions)
    
    corr_optim = torch.optim.Adam(OnlineDecisionTransformer.transformer.params())
    entr_optim = torch.optim.Adam([OnlineDecisionTransformer.temperature])
    
    #Offline Training
    for i in range(10):
        OnlineDecisionTransformer.train(100, 10, corr_optim, entr_optim)
    
    #Online Finetuning
    for i in range(10):
        OnlineDecisionTransformer.sample_traj(0, [DebugEnv])

    OnlineDecisionTransformer.train(100, 10, corr_optim, entr_optim)
    

if __name__ == "__main__":
    unit_test()