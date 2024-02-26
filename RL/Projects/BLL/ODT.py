import transformer
import torch
from copy import deepcopy

class ReplayBuffer:
    def __init__(self, max_length, gamma) -> None:
        self.max_length = max_length
        self.gamma = gamma
        self.content = []

    #Randomly samples from trajectories according to relative size, and then from length-K subtrajectories from the chosen trajectory
    def retrieve(self, K):
        pass
    
    def hindsight_return_labeling(self, traj):
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
            self.content[-(len(self.content) - self.max_length):] = [1]
        
    def add_online(self, traj):
        self.content.append(self.hindsight_return_labeling(traj))
        self.content.pop(0)


class ODT:
    def __init__(self) -> None:
        pass
    
    def pred(self, R, s, a, t):
        pos_embedding = embed_t(t)


    def train_online(self):
        pass

    def train_offline(self):
        pass
