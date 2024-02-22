import numpy as np
import torch

class PPO:
    def __init__(self, model_pol, model_val, env) -> None:
        self.policy = model_pol
        self.val = model_val
        self.env = env
        self.traj = []
        self.adv = 0

    def collect_traj(self, episodes):
        for i in range(episodes):
            while self.env.Terminated != 0:
                decision = self.policy.pred(self.env.State)
                self.traj.append((self.env.state, decision, self.env.reward, self.env.new_state))

    def ppo_step(self):
        self.advantage

class BPPO:
    def ppo_offline_step(self):
        
