import numpy as np
import torch


class NextStep:
    """Carries out the environment steps and adds experiences to memory"""

    def __init__(self, env, agent, replay_buffer, arm):
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.obs = env.reset(arm)
        self.done = False
        self.arm = arm
        self.observation_steps = 50  # 200, need to change average to match in train.py

    def next_step(self, episode_timesteps, noise=0.1):
        action = self.agent.select_action(np.array(self.obs))

        # Perform action
        if self.arm == "left":
            raw_state, raw_next_state, reward, done = self.env.step_left(action)
        else:
            raw_state, raw_next_state, reward, done = self.env.right_left(action)

        if episode_timesteps + 1 == self.observation_steps:
            done_bool = 0
            done = True
        else:
            done_bool = float(done)
        state = list(raw_state.values())
        next_state = list(raw_next_state.values())

        # Store data in replay buffer
        self.replay_buffer.add(raw_state, raw_next_state, action, reward, done_bool)

        self.obs = next_state

        if done:
            # TODO change to random state
            self.obs = self.env.reset_random(self.arm)
            # moves arm to a random starting position
            self.done = False

            return reward, True

        return reward, done

