import torch
import numpy as np
from baxter_init import Baxter

# Configuration variables
from evaluate_policy import evaluate_policy
from observe import observe
from replay_buffer import ReplayBuffer
from step import Step
from td3 import TD3
from train import train

SEED = 0
OBSERVATION = 10000
EXPLORATION = 5000000
BATCH_SIZE = 100
GAMMA = 0.99
TAU = 0.005
NOISE = 0.2
NOISE_CLIP = 0.5
EXPLORE_NOISE = 0.1
POLICY_FREQUENCY = 2
EVAL_FREQUENCY = 5000
REWARD_THRESH = 8000
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
observe(baxter, replay_buffer, OBSERVATION)


# Train agent
train(policy, baxter, REWARD_THRESH, BATCH_SIZE, GAMMA, TAU, NOISE, NOISE_CLIP, POLICY_FREQUENCY, EXPLORATION, replay_buffer)


policy.load()

for i in range(100):
    evaluate_policy(policy, baxter, render=False)

baxter.close_env()
