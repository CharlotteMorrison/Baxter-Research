import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from step import Step


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

    writer = SummaryWriter(comment="-TD3_Baxter")

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

                if avg_reward >= REWARD_THRESH:
                    break

                agent.train(replay_buffer, episode_timesteps, BATCH_SIZE, GAMMA, TAU, NOISE, NOISE_CLIP,
                            POLICY_FREQUENCY)

                # Evaluate episode
                #                 if timesteps_since_eval >= EVAL_FREQUENCY:
                #                     timesteps_since_eval %= EVAL_FREQUENCY
                #                     eval_reward = evaluate_policy(agent, test_env)
                #                     evaluations.append(avg_reward)
                #                     writer.add_scalar("eval_reward", eval_reward, total_timesteps)

                #                     if best_avg < eval_reward:
                #                         best_avg = eval_reward
                #                         print("saving best model....\n")
                #                         agent.save("best_avg","saves")

                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

        reward, done = step.next_step(episode_timesteps)
        episode_reward += reward

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
