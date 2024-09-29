from stable_baselines3 import PPO
import gymnasium as gym
import customEnv

env = gym.make("customEnv/GridWorld-v0", size=10)
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("ppo_vision_game")