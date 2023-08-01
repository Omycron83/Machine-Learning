import numpy as np
import NN

class DDQN:
    class replay_memory:
        def __init__(self, input_size, ouput_size, max_replay_size, min_replay_size):
            self.Input_history = np.zeros((1, input_size)) #s_t
            self.Resulting_state_history = np.zeros((1, input_size)) #s_t+1
            self.Reward_Action_history = np.zeros((1, 2)) #r_t & a_t
            self.max_replay_size = max_replay_size
            self.min_replay_size = min_replay_size
            self.output_size = ouput_size
            self.not_filled = True
        
        def update_history(self, history):
            for k in range(len(history)):
                #Adding the inputs of this iteration to our s_t history:
                self.Input_history = np.vstack((self.Input_history, history[k][0].reshape(1, history[k][0].size)))
                #Adding the resulting state to our s_t+1 - history if are at a non-terminal state, otherwise all elements are null (as put into the buffer)
                self.Resulting_state_history = np.vstack((self.Resulting_state_history, history[k][1]))
                #Adding the reward and action to our r_t & a_t - history:
                self.Reward_Action_history = np.vstack((self.Reward_Action_history, np.array([history[k][2], history[k][3]])))

            if self.not_filled:
                self.not_filled = False
                self.Input_history = np.delete(self.Input_history, 0, 0)
                self.Resulting_state_history = np.delete(self.Resulting_state_history, 0, 0)
                self.Reward_Action_history =  np.delete(self.Reward_Action_history, 0, 0)

            #If our replay buffer is full, we pop the first element of our histories
            if self.Input_history.shape[0] > self.max_replay_size:
                for i in range(self.Input_history.shape[0] - self.max_replay_size):
                    self.Input_history = np.delete(self.Input_history, 0, 0)
                    self.Resulting_state_history = np.delete(self.Resulting_state_history, 0, 0)
                    self.Reward_Action_history =  np.delete(self.Reward_Action_history, 0, 0)
            #print(self.Resulting_state_history, self.Resulting_state_history.shape, np.isnan(self.Resulting_state_history[-1, 0]))

        def DDQN_training(self, learning_rate, batchsize, discount_rate, target_network, update_network):
            #First of all, we want to shuffle our experience replay randomly (with all transitions shuffled accordingly):
            p = np.random.permutation(self.Input_history.shape[0])[:batchsize]
            inputs_shuffled = self.Input_history[p]
            resulting_states_shuffled = self.Resulting_state_history[p]
            rewards_action_shuffled = self.Reward_Action_history[p]

            #We use the bellman equation, thus making the "right" value: r + discount_rate * Q_current(s', argmax Q_target(s', a')), according to Hassel et al, 2015
            #If we arent at the terminal state, we use this temporal difference reward, otherwise just r

            #We basically use stochastic gradient descent for this, since we need to figure out if a given state is terminal
            cost = 0
            for i in range(inputs_shuffled.shape[0]):
                #We use a "q-array", where the only non-zero q-value is the "target" for the chosen q-value
                target_reward = np.zeros((self.output_size))
                if np.isnan(resulting_states_shuffled[i, 0]):
                    target_reward[int(rewards_action_shuffled[i, 1])] = rewards_action_shuffled[i, 0] #The actual reward is just r
                else:
                    update_network.forward_propagation(resulting_states_shuffled[i, :].reshape(1, resulting_states_shuffled.shape[1]), np.zeros((1, update_network.len_output)), NN.MSE)
                    target_network.forward_propagation(resulting_states_shuffled[i, :].reshape(1, resulting_states_shuffled.shape[1]), np.zeros((1, update_network.len_output)), NN.MSE)
                    argmax_target = int(np.where(target_network.layers_for[-1].z[0] == max(target_network.layers_for[-1].z[0]))[0])
                    target_reward[int(rewards_action_shuffled[i, 1])] = rewards_action_shuffled[i, 0] + discount_rate * update_network.layers_for[-1].z[0, int(argmax_target)]
                
                cost += update_network.dqn_gradient_descent(inputs_shuffled[i].reshape(1, resulting_states_shuffled.shape[1]), target_reward, NN.MSE, alpha = learning_rate) 
            

            return cost / batchsize
            

    def __init__(self, env, layers, alpha, batch_size, obs_space, action_space, min_replay, max_replay, discount, update_freq, disp_freq, epsilon, needed_reward, gamma = 0.99, train_freq = 3, episodes_solved = "", terminal_reward = 0):
        #An environment needs the following methods: reset(self-explanatory), render(returns rendered array) and step(takes in an action, returns new state, reward, terminated and truncated)
        self.env = env
        #The target network is the network that is being copied every so often and that gives the target q values for training
        self.target_network = NN.cont_feedforward_nn(obs_space, layers, NN.ReLU, NN.ReLUDeriv, NN.output, NN.MSE_out_deriv, action_space)
        #The online network is the network being updated each step and the one making decisions on each step
        self.online_network = NN.cont_feedforward_nn(obs_space, layers, NN.ReLU, NN.ReLUDeriv, NN.output, NN.MSE_out_deriv, action_space)
        self.memory = self.replay_memory(obs_space, action_space, max_replay, min_replay)
        self.alpha = alpha
        self.discount = discount
        self.update_freq = update_freq
        self.disp_freq = disp_freq
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.needed_reward = needed_reward
        self.train_freq = train_freq
        self.terminal_reward = terminal_reward
        if episodes_solved == "":
            self.episodes_solved = self.disp_freq
        else:
            self.episodes_solved = episodes_solved

    def pick_action(self, output):
        if np.random.randint(0, 101)/100 < 1 - self.epsilon:
            return int(np.where(output == max(output))[0])
        else:
            return np.random.randint(0, self.memory.output_size)

    def test_environment(self):
        observation, info = self.env.reset()
        terminated, truncated = False, False
        reward_acc = 0
        while not (terminated or truncated):
            self.online_network.forward_propagation(observation.reshape(1, observation.size), np.zeros((1, self.online_network.len_output)), NN.MSE)
            output = self.online_network.layers_for[-1].z[0]
            action = int(np.where(output == max(output))[0])
            observation, reward, terminated, truncated, info = self.env.step(action)
            reward_acc += reward
        self.env.reset()
        return reward_acc

    def train_environment(self, iterations):
        print("Training the model: ...")
        observation, info = self.env.reset()
        observation = np.array(observation).reshape(1, observation.size)
        self.online_network.forward_propagation(observation.reshape(1, observation.size), np.zeros((1, self.online_network.len_output)), NN.MSE)
        output = self.online_network.layers_for[-1].z[0]

        reward_acc = 0
        #To keep track of the accumulated reward in each episode
        reward_acc_episode = 0
        #History of those rewards
        self.reward_intermediate_history = []
        #History of the test rewards
        self.reward_test_history = []
        error = 0
        self.play_throughs = 0
        while self.play_throughs < iterations:
            #Epsilon decay with some value gamma
            if self.epsilon > 0.05:
                self.epsilon *= self.gamma
            
            #Choosing an action
            action = self.pick_action(output)
            
            #Undergoing a transition in the environment with that action, where we enter a_t and receive s_t+1, r_t and if we have entered a terminal state
            observation_new, reward, terminated, truncated, info = self.env.step(action)

            #Reshaping our new observation
            observation_new = np.array(observation_new).reshape(1, observation.size)
            
            #In order to keep track of average rewards
            reward_acc_episode += reward

            #If we have entered a terminal state, s_t+1 is NaN to signify that we have entered a terminal state, and we reset the environment to get the first state of the next episode
            if terminated or truncated:
                reward += self.terminal_reward #: Optional reward at termination, not necessary if reward structure is reasonable
                observation_new = observation_new.astype(np.float64)
                observation_new[:] = np.nan
                self.memory.update_history([[observation, observation_new, reward, action]])
                observation, info = self.env.reset()
            #otherwise, s_t+1 is just observation_new and the observation 
            else:
                self.memory.update_history([[observation, observation_new, reward, action]])
                observation = observation_new

            #Training our online network from experience replay at every step
            if self.memory.Input_history.shape[0] > self.memory.min_replay_size:
                for i in range(self.train_freq):
                    error = self.memory.DDQN_training(self.alpha, self.batch_size, self.discount, self.target_network, self.online_network) / 3
            
            #Now that we have this new observation, lets calculate the q function at this state
            self.online_network.forward_propagation(observation.reshape(1, observation.size), np.zeros((1, self.online_network.len_output)), NN.MSE)
            output = self.online_network.layers_for[-1].z[0]
            
            #When an episode is finished, we sometimes copy weights/visualize results
            if terminated or truncated:
                self.reward_intermediate_history.append(reward_acc_episode)
                if self.play_throughs % self.update_freq == 0:
                    import pathlib
                    path = str(pathlib.Path(__file__).parent.resolve()) + "weights.pkl"
                    weights = self.online_network.retrieve_weights()
                    self.target_network.assign_weights(weights)
                    self.target_network.save_weights(path)
                
                x = self.test_environment()
                self.reward_test_history.append(x)

                reward_acc = 0
                if len(self.reward_intermediate_history) != 1:
                    for i in range(1, min([self.episodes_solved + 1, len(self.reward_intermediate_history)])):
                        reward_acc += self.reward_intermediate_history[-i] / min([self.episodes_solved, len(self.reward_intermediate_history)])
                else:
                    reward_acc += self.reward_intermediate_history[-1]

                #If we "solve" the environment, we might want to look at the results first.
                if reward_acc > self.needed_reward and x > self.needed_reward:
                    print("Prematurely ended with an average result of", reward_acc, "and an latest reward of", x, "at epsiode", self.play_throughs, ".")
                    break

                if self.play_throughs % self.disp_freq == 0:
                    print("At episode", self.play_throughs,"cost:", error, "current reward:", x, "average accumulated reward:", reward_acc, ".")
                
                reward_acc_episode = 0
                self.play_throughs += 1

    def visualize(self, env_render, path, name):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        print("Visualizing the training process: ...")
        plt.plot(self.reward_intermediate_history, color = "r", label = "Intermediate")
        plt.plot(self.reward_test_history, color = "b", label = "Test")
        plt.xlabel("Epoch")
        plt.ylabel("Reward")
        plt.legend()
        plt.show()

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
        self.online_network.forward_propagation(observation.reshape(1, observation.size), np.zeros((1, self.online_network.len_output)), NN.MSE)
        output = self.online_network.layers_for[-1].z[0]
        frames = []
        reward_acc = 0
        while True:
            action = int(np.where(output == max(output))[0])
            observation, reward, terminated, truncated, info = env_render.step(action)
            reward_acc += reward
            observation = np.array(observation).reshape(1, observation.size)
            self.online_network.forward_propagation(observation.reshape(1, observation.size), np.zeros((1, self.online_network.len_output)), NN.logistic_cost)
            output = self.online_network.layers_for[-1].z[0]
            frames.append(env_render.render()) #Or however you want to retrieve stuff
            if terminated or truncated:
                print("Visualization reward of", reward_acc)
                break
        env_render.close()
        save_frames_as_gif(frames)

    def info(self):
        print("Parameters: env, nodes, alpha, batchsize, input_len, output_len, min_replay, max_replay, discount, update_freq, disp_freq, epsilon, stop_reward, (gamma), (train_freq), (episodes_solved), (terminal reward)")
        print("Of this instance:", self.env, self.target_network.nodes, self.alpha, self.batch_size, "...", "...", self.memory.min_replay_size, self.memory.max_replay_size, self.discount, self.update_freq, self.epsilon, self.needed_reward, self.gamma, self.train_freq, self.episodes_solved, self.terminal_reward)