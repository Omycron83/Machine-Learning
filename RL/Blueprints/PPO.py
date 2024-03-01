import numpy as np
import torch
from copy import deepcopy

class PPO:
    def __init__(self, model_pol, model_val, env, gamma) -> None:
        self.policy = model_pol
        self.val = model_val
        self.env = env
        self.gamma = gamma
        #trajectory
        self.traj = []
        self.adv = 0

    def hindsight_return_labeling(self, traj_list):
        #Making a deep-copy as to mitigate external errors in reuse trajectories
        traj_list = deepcopy(traj_list)
        #Updating trajectory return-to-go as discounted sum in hindsight return labeling in monte-carlo fashion
        for j in range(len(traj_list)):
            for i in range(1, len(traj_list[j]) + 1):
                if i != 1:
                    traj_list[j][-i][0] += self.gamma * traj_list[j][-(i - 1)][0]
        return traj_list
    
    def collect_traj(self, episodes):
        #Obtain a number of episodes using the current stochastic policy
        for i in range(episodes):
            while self.env.Terminated != 0:
                decision = self.policy.pred(self.env.State)
                self.traj.append((self.env.state, decision, self.env.reward, self.env.new_state))
        self.traj = self.hindsight_return_labeling(self.traj)

    #Optains a PPO-Step, where 
    def ppo_step(self):
        self.advantage = torch.cat
        model_params = self.policy.model_params
        self.policy.pred(states)
        loss = torch.minimum()

class BPPO:
    def ppo_offline_step(self):
        
