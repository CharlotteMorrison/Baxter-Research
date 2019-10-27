import numpy as np
import torch

from utility.baxter_init import Baxter
from td3algorithm.evaluate_policy import evaluate_policy
from td3algorithm.observe import observe
from td3algorithm.replay_buffer import ReplayBuffer
from td3algorithm.next_step import NextStep
from td3algorithm.td3 import TD3
from td3algorithm.train import train


if __name__ == '__main__':
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
    REWARD_THRESH = 11.95  # 8000
    # initialize the baxter environment
    baxter = Baxter()

    # set arms at neutral position
    baxter.neutral()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    state_dim = baxter.observation_space()
    state_dim = len(state_dim[1])
    action_dim = len(baxter.action_space())

    action = baxter.observation_space()

    policy = TD3(state_dim, action_dim, action, device, baxter)

    replay_buffer = ReplayBuffer()

    step = NextStep(baxter, policy, replay_buffer, "left")

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True

    # once replay buffer is full, use the pre-made one to populate observe step

    # Populate replay buffer
    observe(baxter, replay_buffer, OBSERVATION, "left")

    # Train agent
    train(policy, baxter, REWARD_THRESH, BATCH_SIZE, GAMMA, TAU, NOISE, NOISE_CLIP, POLICY_FREQUENCY, EXPLORATION,
          replay_buffer, step, "left")

    policy.load()

    for i in range(100):
        evaluate_policy(policy, baxter)

    baxter.close_env()
