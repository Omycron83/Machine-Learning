import gym
import DoubleDQN
x = DoubleDQN.DDQN(gym.make('CartPole-v1'), [32, 32], 0.0004, 32, 4, 2, 0, 100_000, 0.95, 1, 10, 1, 490)
x.info()
x.train_environment(200)
x.visualize(gym.make('CartPole-v1', render_mode = "rgb_array"), "D:\Damian\PC\Python\ML\RL\Projects\OpenAIGym\Stick\Visualization\ ", "stick.gif")