import transformer_improved
import torch
from copy import deepcopy
import random

#A replay buffer saving trajectories of the form (g_t, s_t, a_t) in a list
class ReplayBuffer:
    def __init__(self, max_length, gamma) -> None:
        self.max_length = max_length
        self.gamma = gamma
        self.content = []

    #Randomly samples from trajectories according to relative size, and then from length-K subtrajectories from the chosen trajectory
    def retrieve(self, K):
        traj_lengths = [len(k) for k in self.content]
        chosen_traj = random.choices(self.content, weights=traj_lengths)[0]
        if len(chosen_traj) > K:
            slice_index = random.randint(0, len(chosen_traj) - K)
            return chosen_traj[slice_index:K + slice_index]
        else:
            return chosen_traj
    
    def hindsight_return_labeling(self, traj_list):
        #Making a deep-copy as to mitigate external errors in reuse trajectories
        traj_list = deepcopy(traj_list)
        #Updating trajectory return-to-go as discounted sum in hindsight return labeling in monte-carlo fashion
        for j in range(len(traj_list)):
            for i in range(1, len(traj_list[j]) + 1):
                if i != 1:
                    traj_list[j][-i][0] += self.gamma * traj_list[j][-(i - 1)][0]
        return traj_list

    #Assumes a list of trajectories, given each by a sublist of steps, i.e. tuples of the form (r, s ,a)
    def add_offline(self, traj_list):
        traj_list = self.hindsight_return_labeling(traj_list)
        #Adding augmented trajectories to saved 
        self.content += traj_list

        #Sorting the list and deleting 'over-the-top' elements
        self.content.sort(key = lambda trajectory: trajectory[0][0])
        if len(self.content) > self.max_length:
            self.content[-(len(self.content) - self.max_length):] = []
        
    def add_online(self, traj):
        self.content.append(self.hindsight_return_labeling(traj))
        if len(self.content) > self.max_length:
            self.content.pop(0)

#Used in continuous action spaces
class OutputDist(transformer.OutputLayer):
    def forward(self, x):
        return 

#Used in discrete action spaces


class ODTEmb(transformer.EmbeddingLayer):
    def forward(self, x):
        return 

#High-level class implementing the ODT-algorithm using multiple other classes to enable rl training
class ODT:
    #As a GPT-Architecture is used, we 
    def __init__(self, d_input: int, d_output: int, d_model: int, n_head: int, dim_ann: int, dec_layers: int, context_length: int, gamma: float, lr: float, env):
        #Initializes the function approximator in this type of task
        self.transformer = transformer.Transformer(d_input, d_output, d_model, n_head, dim_ann, ODTEmb, 0, dec_layers, OutputDist)

        self.ReplayBuffer = ReplayBuffer(context_length, gamma)

        #Warm-up not needed, but might try it out later on
        self.AdamW = torch.optim.AdamW(lr = lr)

        #The current online-trajectory being encountered. This 
        self.current_traj = []

        #The environment acted on, has to implement:
        """
        env: performs a step in the environment; env.step(action) -> s_t+1, r_t, terminated
        reset: resets the environment data
        is_terminated: 
        """
        self.env = env


    #Optimization algorithm: AdamW
    def train(self, iterations, lr, batch_size):
        #Retrieving data from replay buffer, make this be vectorized
        replays = []
        for i in range(batch_size):
            replays.append(self.ReplayBuffer.retrieve())
        replays = torch.cat(replays, dim=2)

        #Predicting outputs, getting error and 
        self.AdamW.zero_grad()
        output = self.transformer.forward(replays)
        self.

    def env_step(self, current_traj):
        output = self.transformer.forward(current_traj)
        

    def add_data(self, traj_list):
        self.ReplayBuffer.add_offline(traj_list)
        