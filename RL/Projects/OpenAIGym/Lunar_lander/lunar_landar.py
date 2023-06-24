import gym
import DoubleDQN
x = DoubleDQN.DDQN(gym.make("LunarLander-v2"), [256, 256], 0.00008, 32, 8, 4, 0, 1_000_000, 0.99, 10, 10, 1, 200, gamma = 0.92)
x.info()
x.train_environment(4000)
x.visualize(gym.make("LunarLander-v2", render_mode = "rgb_array"), "D:\Damian\PC\Python\ML\RL\Projects\OpenAIGym\Lunar_lander\Visualization\ ", "lander.gif")