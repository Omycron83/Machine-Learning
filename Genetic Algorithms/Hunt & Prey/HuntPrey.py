#here, we want to simulate evolution of hunter and prey through a genetic algorithm
#in this algorithm, NN-Weights are supposed to be explored and learned, with no hidden layers
#they will map inputs from the line of sight of both hunter and prey
#(hunter l-o-s is further but more far, prey l-o-s shorter but basically 360 degree's, see png)
#to a movement sigmoid and orientation softmax (0, 1, 2, 3) vector
import numpy as np

class Hunter:
    def sigmoid(z):
        return 1. /(1 + np.exp(z))
    def softmax(z):
        return np.exp(z) / (np.sum(np.exp(z)))
    
    def __init__(self, pos, orient):
        self.pos = pos
        self.orient = orient
        self.theta_mov = np.zeros((2,))
        self.theta_orient = np.zeros((2, 4))
        self.theta_sprint = np.zeros((3, ))
        self.energy = 100
    def forprop(self, rays):
        mov = self.sigmoid(rays @ self.theta_mov) > 0.5
        sprint = self.sigmoid(np.hstack((np.array[self.energy].reshape(1,1), rays.reshape(2, 1))) @ theta_sprint) > 0.5
        orient = np.argmax(self.softmax(rays @ self.theta_orient))
        return mov, orient, sprint
    def each_step(self, rays, positions):
        mov, orient, sprint = forprop(rays)
        self.energy -= 5 + sprint * 10
        #positions = [[2, 1], [5, 2]
        #             [4, 2], [2, 1]]
                     
        
    
#simulation parameters:
x = 1200
y = 1200
hunters = 50
prey = 50

def simulation(field, hunters, prey):
    
