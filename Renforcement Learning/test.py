import gymnasium as gymn

env = gymn.make("CartPole-v1",render_mode="human")
observation , info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
