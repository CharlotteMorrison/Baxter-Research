import numpy as np
import torch


class Step:
    """Carries out the environment steps and adds experiences to memory"""

    def __init__(self, env, agent, replay_buffer, arm):
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.obs = env.reset(arm)
        self.done = False
        self.arm = arm
        self.observation_steps = 200

    def next_step(self, episode_timesteps, noise=0.1):
        action = self.agent.select_action(np.array(self.obs), noise=0.1)

        # Perform action
        raw_state, raw_next_state, reward, done = self.env.step(self.arm, action)
        done_bool = 0 if episode_timesteps + 1 == self.observation_steps else float(done)
        state = list(raw_state.values())
        next_state = list(raw_next_state.values())
        # Store data in replay buffer
        self.replay_buffer.add(state, next_state, action, reward, done_bool)

        self.obs = next_state

        if done:
            self.obs = self.env.reset(self.arm)
            done = False

            return reward, True

        return reward, done
