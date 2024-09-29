import gymnasium as gym
from stable_baselines3 import PPO
import customEnv  # Ensure this imports your custom environment correctly

# Load the trained model
model = PPO.load("ppo_vision_game")

# Create the environment
env = gym.make("customEnv/GridWorld-v0", size=10, render_mode="human")

# Number of episodes to evaluate
num_episodes = 10

# Evaluate the model
episode_rewards = []
for episode in range(num_episodes):
    obs, info = env.reset()  # Unpack the tuple returned by reset
    done = False
    truncated = False
    total_reward = 0
    while not (done or truncated):
        action, _states = model.predict(obs)
        action = int(action)  # Convert the action to an integer
        obs, reward, done, truncated, info = env.step(action)  # Unpack the five-element tuple
        total_reward += reward
    episode_rewards.append(total_reward)

# Print the results
for i, reward in enumerate(episode_rewards):
    print(f"Episode {i+1}: Total Reward: {reward}")

print(f"Average Reward over {num_episodes} episodes: {sum(episode_rewards) / num_episodes}")
