import snake_env
import DoubleDQN
x = DoubleDQN.DDQN(snake_env.snake_game_pixel(20, 20), [512, 512], 0.00001, 32, 400, 4, 0, 1_000_000, 0.95, 5, 10, 10, 40, gamma = 0.995, train_freq = 3)
x.info()
x.train_environment(800)
x.visualize(snake_env.snake_game_pixel(20, 20), "D:\Damian\PC\Python\ML\RL\Projects\RL_Snake\Visualization\ ", "snake.gif")