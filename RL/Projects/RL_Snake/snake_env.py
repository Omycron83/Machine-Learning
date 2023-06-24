import numpy as np
import pylab as plt
from random import randint
import NN
import winsound

class snake_game_pixel:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.field = np.zeros((x, y))
        self.snake = [[y//2, x//2], [y//2, x//2-1], [y//2, x//2 - 2], [y//2, x//2 - 3]]
        self.apple = [np.random.randint(0, y), np.random.randint(0, x)]
        self.inputs = [0]*len(self.snake)
        self.steps_since_last_apple = 0
    
    def update_field(self):
        new_field = np.zeros((self.x, self.y))
        color = 255
        for i in self.snake:
            new_field[i[0], i[1]] += color
            color -= 5
        new_field[self.apple[0], self.apple[1]] += 100
        self.field = new_field

    def reset(self):
        self.field = np.zeros((self.x, self.y))
        self.snake = [[self.y//2, self.x//2], [self.y//2, self.x//2-1], [self.y//2, self.x//2 - 2], [self.y//2, self.x//2 - 3]]
        self.apple = [np.random.randint(0, self.y), np.random.randint(0, self.x)]
        self.inputs = [0]*len(self.snake)
        self.steps_since_last_apple = 0
        self.update_field()
        return self.get_obs(), ""
    
    def get_obs(self):
        return self.render()

    def render(self):
        return self.field
    
    def close(self):
        del self

    def step(self, action):
        reward = 0
        last = self.inputs.pop()
        self.inputs.insert(0, action)
        oldDistance = np.sqrt((self.snake[0][0] - self.apple[0])**2 + (self.snake[0][1] - self.apple[1])**2)
        for i in range(len(self.snake)):
            self.snake[i][0] += (self.inputs[i] == 2)*(-1) + (self.inputs[i] == 3)*(1) #2: up, 3: down
            self.snake[i][1] += (self.inputs[i] == 0)*(1) + (self.inputs[i] == 1)*(-1) #0: right, 1: left
        newDistance = np.sqrt((self.snake[0][0] - self.apple[0])**2 + (self.snake[0][1] - self.apple[1])**2)
        reward += 0.1 * (oldDistance - newDistance)
        #We want to de-incentivize spinning around in an endless loop
        #If that doesnt work, we will just let it "starve":
        if self.steps_since_last_apple >= 300:
            reward -= 100
            return self.get_obs(), reward, False, True, ""
        for i in self.snake:
            if i == self.apple:
                self.snake.append([self.snake[-1][0] + (self.inputs[-1] == 2)*(1) + (self.inputs[-1] == 3)*(-1), self.snake[-1][1] + (self.inputs[-1] == 0)*(-1) + (self.inputs[-1] == 1)*(1)])
                self.inputs.append(last)
                self.apple = [np.random.randint(0, self.y), np.random.randint(0, self.x)]
                reward += 20
                self.steps_since_last_apple = 0
            if self.snake.count(i) > 1 or i[0] >= self.y or i[0] < 0 or i[1] >= self.x or i[1] < 0:
                reward -= 20
                return self.get_obs(), reward, True, False, ""
            else:
                terminated = False
        self.steps_since_last_apple += 1
        self.update_field()
        return self.get_obs(), reward, terminated, False, ""

class snake_game_nopixel(snake_game_pixel):
    #This apparently still needs some tweaking!
    def get_obs(self):
        output = []
        for i in self.apple:
            for j in self.snake[0]:
                if j > i:
                    output.append(1)
                else:
                    output.append(0)
        if self.snake[0][0] + 1 < 0 or self.snake[0][0] + 1 > 19 or self.snake.count([self.snake[0][0] + 1, self.snake[0][1]]) > 1:
            output.append(1)
        else:
            output.append(0)
        if self.snake[0][0] - 1 < 0 or self.snake[0][0] - 1 > 19 or self.snake.count([self.snake[0][0] - 1, self.snake[0][1]]) > 1:
            output.append(1)
        else:
            output.append(0)
        if self.snake[0][1] + 1 < 0 or self.snake[0][1] + 1 > 19 or self.snake.count([self.snake[0][0], self.snake[0][1] + 1]) > 1:
            output.append(1)
        else:
            output.append(0)
        if self.snake[0][1] - 1 < 0 or self.snake[0][1] - 1 > 19 or self.snake.count([self.snake[0][0], self.snake[0][1] - 1]) > 1:
            output.append(1)
        else:
            output.append(0)
        output = np.array(output)
        return output