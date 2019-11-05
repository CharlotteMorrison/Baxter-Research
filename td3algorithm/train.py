import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from subplot import Subplot


def train(agent, env, replay_buffer, step, arm):
    """Train the agent for exploration steps

        Args:
            :param step: (NextStep)  move to the next step
            :param replay_buffer: (ReplayBuffer) replay buffer for arm
            :param agent: (Agent): agent to use
            :param env: (environment): gym environment
            :param arm: (string): "left" or "right"
    """
    EXPLORATION = 5000000
    REWARD_THRESH = 1.95  # 8000

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    done = False
    rewards = []
    best_avg = -2

    writer = SummaryWriter(comment="TD3_Baxter")
    avg_reward_plot = Subplot()
    best_reward_plot = Subplot()

    while total_timesteps < EXPLORATION:
        if done:
            if total_timesteps != 0:
                rewards.append(episode_reward/episode_timesteps)
                avg_reward = np.mean(rewards[-10:])

                # graph the average/best rewards
                avg_reward_plot.reward_subplot(avg_reward, "Current_Average_Reward")
                best_reward_plot.reward_subplot(best_avg, "Best_Average_Reward")

                writer.add_scalar("avg_reward", avg_reward, total_timesteps)
                writer.add_scalar("reward_step", reward, total_timesteps)
                writer.add_scalar("episode_reward", episode_reward, total_timesteps)

                # print("Average reward: " + str(avg_reward) + "  Best Reward: " + str(best_avg))
                if best_avg < avg_reward:
                    best_avg = avg_reward
                    print("saving best model....\n")
                    agent.save("best_avg", "saves")

                print("\rTotal T: {:d} Episode Num: {:d} Reward: {:f} Avg Reward: {:f}".format(
                    total_timesteps, episode_num, episode_reward, avg_reward))
                sys.stdout.flush()

                if avg_reward >= REWARD_THRESH:
                    break

                # trains with the TD3 function
                agent.train(replay_buffer)

                # reset the values for a new episode, increment number of episodes
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

        # run a new step
        reward, done = step.next_step(episode_timesteps)
        # add the reward to the episode total for later averaging
        episode_reward += reward

        # increment the timesteps
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
        print("Total Timesteps: " + str(total_timesteps) + "   Reward: " + str(reward) + "   Done: " + str(done))
