import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import shutil
import time
from time import sleep
import numpy as np
import torch
import torch.nn.functional as F
import math
import random
from snakeGame import SnakeEnv


is_cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda_available else "cpu")


dqn_net = torch.load("./models_DQN/model_DQN.pth", device)
dqn_net.eval()


def sample_action(obs, random_rate=0.0):
    if random.random() < random_rate:
        action_ind = random.randint(0, 3)
    else:
        snake_state = torch.tensor(obs[0]).view(-1, 1, 100).to(device)
        food_state = torch.tensor(obs[1]).view(-1, 1, 100).to(device)
        quality, value = dqn_net(snake_state, food_state)
        quality: torch.Tensor = quality.view(-1, 4)
        print([f"{q:7.3f}" for q in quality.detach().numpy().flatten()])
        action_ind = quality.argmax().item()
    return action_ind


env = SnakeEnv()
obs = env.reset()
done = False
step = 0
while not done:
    step += 1
    action = sample_action(obs=obs)
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.0005)

print(f"step: {step}")
os.system("pause")