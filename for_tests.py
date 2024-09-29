import gymnasium
import customEnv

env = gymnasium.make("customEnv/GridWorld-v0",render_mode="human" , size=10)

state = env.reset()

# Run a few steps in the environment
for _ in range(100):
    action = env.action_space.sample()  # Sample a random action
    state, reward, done,truncated, info = env.step(action)
    env.render()  # Render the current state in human mode
    if done:
        break

env.close()

