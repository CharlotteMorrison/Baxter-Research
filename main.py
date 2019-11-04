import numpy as np
import torch
import cPickle as pickle

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
    EXPLORE_NOISE = 0.1

    # TD3 hyperparameters from addressing function approx. err paper
    gamma = 0.99   # discount
    tau = 0.005
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2
    # initialize the baxter environment
    baxter = Baxter()

    # set arms at neutral position
    baxter.neutral()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    state_dim = baxter.observation_space()
    # state_dim = len(state_dim[1])
    action_dim = baxter.action_space()

    agent = TD3(state_dim, action_dim, action_dim[0], gamma, tau, policy_noise, noise_clip, policy_freq, device)

    replay_buffer = ReplayBuffer()

    step = NextStep(baxter, agent, replay_buffer, "left")

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True

    observe(baxter, replay_buffer, OBSERVATION, "left")

    # Train agent
    train(agent, baxter, replay_buffer, step, "left")

    agent.load()

    for i in range(100):
        # need to pass arm, add arm to evaluate policy
        evaluate_policy(agent, baxter, "left")

    baxter.close_env()
