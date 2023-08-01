import numpy as np
import NN

#This implementation is fundamentally flawed in the way training is handled, improve asap
class DDQN:
    class replay_memory:
        def __init__(self, input_size, output_size, max_replay_size, min_replay_size):
            self.Input_history = np.zeros((1, input_size))
            self.Output_history = np.zeros((1, output_size))
            self.max_replay_size = max_replay_size
            self.min_replay_size = min_replay_size
        
        def update_history(self, history, discount_rate, update_network):
            for k in range(len(history)):
                #Adding all elements from our history to the replay buffer
                self.Input_history = np.vstack((self.Input_history, history[k][0].reshape(1, history[k][0].size)))
                #We, first of all, initialize the expected reward of our actions to 0, and after that alter the taken action's reward based on td-reward
                self.Output_history = np.vstack((self.Output_history, np.zeros((1, history[k][1].size))))

                #(Naive) Option 1: we just figure out the value of the subsequent actions, and then discount those, i.e. use a bootstrapping method:
                self.Output_history[-1][history[k][3]] = 0
                for l in range(0, len(history) - k):
                    self.Output_history[-1][history[k][3]] += history[k + l][2] * discount_rate**(l) 
        
            #If our replay buffer is full, we pop the first element of our histories
            if self.Input_history.shape[0] > self.max_replay_size:
                for i in range(self.Input_history.shape[0] - self.max_replay_size):
                    self.Input_history = np.delete(self.Input_history, 0, 0)
                    self.Output_history = np.delete(self.Output_history, 0, 0)

    def __init__(self, env, layers, alpha, batch_size, obs_space, action_space, min_replay, max_replay, discount, update_freq, disp_freq, epsilon, needed_reward, gamma = 0.99):
        #An environment needs the following methods: reset(self-explanatory) and step(takes in an action, returns new state, reward, terminated and truncated)
        self.env = env
        self.update_network = NN.cont_feedforward_nn(obs_space, len(layers), layers, NN.ReLU, NN.ReLUDeriv, NN.output, NN.MSE_out_deriv, action_space)
        self.target_network = NN.cont_feedforward_nn(obs_space, len(layers), layers, NN.ReLU, NN.ReLUDeriv, NN.output, NN.MSE_out_deriv, action_space)
        self.memory = self.replay_memory(obs_space, action_space, max_replay, min_replay)
        self.alpha = alpha
        self.discount = discount
        self.update_freq = update_freq
        self.disp_freq = disp_freq
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.needed_reward = needed_reward

    def pick_action(self, output):
        if np.random.randint(0, 101)/100 < 1 - self.epsilon:
            return int(np.where(output == max(output))[0])
        else:
            return np.random.randint(0, self.memory.Output_history.shape[1])

    def test_environment(self):
        observation, info = self.env.reset()
        terminated, truncated = False, False
        reward_acc = 0
        while not (terminated or truncated):
            self.target_network.forward_propagation(observation.reshape(1, observation.size), np.zeros((1, self.target_network.len_output)), NN.MSE)
            output = self.target_network.layers_for[-1].z[0]
            action = int(np.where(output == max(output))[0])
            observation, reward, terminated, truncated, info = self.env.step(action)
            reward_acc += reward
        self.env.reset()
        return reward_acc

    def train_environment(self, iterations):
        history = []

        observation, info = self.env.reset()
        observation = np.array(observation).reshape(1, observation.size)
        self.target_network.forward_propagation(observation.reshape(1, observation.size), np.zeros((1, self.target_network.len_output)), NN.MSE)
        output = self.target_network.layers_for[-1].z[0]

        reward_acc = 0
        error = 0

        play_throughs = 0
        while play_throughs < iterations:
            #Epsilon decay with some value gamma
            if self.epsilon > 0.05:
                self.epsilon *= self.gamma
            
            #Choosing an action
            action = self.pick_action(output)
            
            #Undergoing a transition in the environment with that action, sampling new reward
            observation_new, reward, terminated, truncated, info = self.env.step(action)
            
            #if terminated or truncated:
            #    reward = -20

            history.append([observation, output, reward, action])
            #After updating our history with the state the action has been taken in as well as the reward received, we update the new state
            observation = np.array(observation_new).reshape(1, observation.size)
            
            #Sampling from experience replay and training our network after every step (also could be after every episode)
            if self.memory.Input_history.shape[0] > self.memory.min_replay_size:
                for i in range(3):
                    p = np.random.permutation(self.memory.Input_history.shape[0])
                    batch_X = self.memory.Input_history[p]
                    batch_Y = self.memory.Output_history[p]
                    error = self.update_network.dqn_stochastic_gradient_descent(self.alpha, 0.0, batch_X[:self.batch_size, :], batch_Y[:self.batch_size, :], NN.MSE)
            
            #Now that we have this new observation, lets calculate the q function at this state
            self.target_network.forward_propagation(observation.reshape(1, observation.size), np.zeros((1, self.target_network.len_output)), NN.MSE)
            output = self.target_network.layers_for[-1].z[0]
            
            #When an episode is finished, we update our experience replay, reset the environment, train the model and sometimes copy weights/visualize results
            if terminated or truncated:
                self.memory.update_history(history, self.discount, self.update_network)
                history = []
                observation, info = self.env.reset()

                if play_throughs % self.update_freq == 0:
                    import pathlib
                    path = str(pathlib.Path(__file__).parent.resolve()) + "weights.pkl"
                    weights = self.update_network.retrieve_weights()
                    self.target_network.assign_weights(weights)
                    self.update_network.save_weights(path)
                if play_throughs % self.disp_freq == 0:
                    print("At iteration", play_throughs,"Error:", error, "Reward:", self.test_environment())
                #If we "solve" the environment, we might want to look at the results first.
                x = self.test_environment()
                if x > self.needed_reward:
                    print(x)
                    break
                play_throughs += 1

    def visualize(self, env_render, path, name):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        def save_frames_as_gif(frames, path = path, filename = name):
            #Mess with this to change frame size
            plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

            patch = plt.imshow(frames[0])
            plt.axis('off')

            def animate(i):
                patch.set_data(frames[i])
            writer = animation.PillowWriter(fps=20)
            anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
            anim.save(path + filename, writer=writer)
        
        observation, info = env_render.reset()
        self.target_network.forward_propagation(observation.reshape(1, observation.size), np.zeros((1, self.target_network.len_output)), NN.MSE)
        output = self.target_network.layers_for[-1].z[0]
        frames = []

        while True:
            action = int(np.where(output == max(output))[0])
            observation, reward, terminated, truncated, info = env_render.step(action)
            observation = np.array(observation).reshape(1, observation.size)
            self.target_network.forward_propagation(observation.reshape(1, observation.size), np.zeros((1, self.target_network.len_output)), NN.logistic_cost)
            output = self.target_network.layers_for[-1].z[0]
            frames.append(env_render.render()) #Or however you want to retrieve stuff
            if terminated or truncated:
                break
        env_render.close()
        save_frames_as_gif(frames)