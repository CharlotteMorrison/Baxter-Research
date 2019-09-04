import torch
import numpy as np
from baxter_init import Baxter

# Configuration variables
from replay_buffer import ReplayBuffer
from step import Step
from td3 import TD3

SEED = 0

# initialize the baxter environment
baxter = Baxter()

# set arms at neutral position
baxter.neutral()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)


state_dim = baxter.observation_space()

action_dim = baxter.action_space()

max_action = state_dim[1]

policy = TD3(state_dim, action_dim, baxter)

replay_buffer = ReplayBuffer()

step = Step(baxter, policy, replay_buffer)

total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True

# Populate replay buffer
observe(env, replay_buffer, OBSERVATION)


# Train agent
train(policy, env)


policy.load()

for i in range(100):
    evaluate_policy(policy, env, render=True)

env.close()