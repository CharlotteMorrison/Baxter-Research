import sys


def observe(env, replay_buffer, observation_steps, arm):
    """run episodes while taking random actions and filling replay_buffer

        Args:
            :param env: (env): gym environment
            :param replay_buffer: (ReplayBuffer): buffer to store experience replay
            :param observation_steps: (int): how many steps to observe for
            :param arm: (string): right or left for arm

    """

    time_steps = 0
    obs = env.reset(arm)
    done = False

    while time_steps < observation_steps:
        if arm == "left":
            obs, new_obs, action, reward, done = env.random_step_left()
            print
        else:
            obs, new_obs, action, reward, done = env.random_step_right()
        replay_buffer.add(obs, new_obs, action, reward, done)

        # obs = new_obs
        time_steps += 1

        if done:
            obs = env.reset(arm)
            done = False

        # if time_steps % 10 == 0:
        print("\rPopulating Buffer {}/{}.".format(time_steps, observation_steps))
        sys.stdout.flush()
        # print("\rWriting Buffer: obs {}, new obs {}, action {}, reward {}, done {}".format
        # (obs, new_obs, action, reward, done))
