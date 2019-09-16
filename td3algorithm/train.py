import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from next_step import NextStep
from subplot import reward_subplot


def train(agent, env, REWARD_THRESH, BATCH_SIZE, GAMMA, TAU, NOISE, NOISE_CLIP, POLICY_FREQUENCY, EXPLORATION,
          replay_buffer, step, arm):
    """Train the agent for exploration steps

        Args:
            agent (Agent): agent to use
            env (environment): gym environment
    """

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    done = False
    obs = env.reset(arm)
    evaluations = []
    rewards = []
    best_avg = -2000

    # plot lists
    plot_avg_reward = []
    plot_episode_reward = []
    plot_max_timesteps =[]

    writer = SummaryWriter(comment="TD3_Baxter")

    while total_timesteps < EXPLORATION:

        if done:

            if total_timesteps != 0:
                rewards.append(episode_reward)
                avg_reward = np.mean(rewards[-100:])

                writer.add_scalar("avg_reward", avg_reward, total_timesteps)
                writer.add_scalar("reward_step", reward, total_timesteps)
                writer.add_scalar("episode_reward", episode_reward, total_timesteps)

                if best_avg < avg_reward:
                    best_avg = avg_reward
                    print("saving best model....\n")
                    agent.save("best_avg", "saves")

                print("\rTotal T: {:d} Episode Num: {:d} Reward: {:f} Avg Reward: {:f}".format(
                    total_timesteps, episode_num, episode_reward, avg_reward))
                sys.stdout.flush()

                # plot the reward values
                plot_avg_reward.append([avg_reward, episode_num + 1])
                plot_episode_reward.append([episode_reward, episode_num + 1])
                plot_max_timesteps.append([total_timesteps, episode_num + 1])
                reward_subplot(plot_avg_reward, plot_episode_reward, plot_max_timesteps)

                if avg_reward >= REWARD_THRESH:
                    break

                agent.train(replay_buffer, episode_timesteps, BATCH_SIZE, GAMMA, TAU, NOISE, NOISE_CLIP,
                            POLICY_FREQUENCY)

                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

        reward, done = step.next_step(episode_timesteps)
        episode_reward += reward

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
