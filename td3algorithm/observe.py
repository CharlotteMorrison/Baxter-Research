import sys
import cPickle as pickle


def observe(env, replay_buffer, observation_steps, arm):
    """run episodes while taking random actions and filling replay_buffer

        Args:
            :param env: (env): gym environment
            :param replay_buffer: (ReplayBuffer): buffer to store experience replay
            :param observation_steps: (int): how many steps to observe for
            :param arm: (string): right or left for arm

    """
    time_steps = 0
    env.reset(arm)
    buffer_storage = []

    # chooses the correct arm action, default left
    obs = env.left_state()
    if arm == "right":
        obs = env.right_state()

    while time_steps < observation_steps:
        # gets the new observation
        # deal with the scope issue, this is janky... find a better way later.
        new_obs, action, reward, done = [None, None, 0.0, False]
        if arm == "left":
            new_obs, action, reward, done = env.random_step_left()
        else:
            new_obs, action, reward, done = env.random_step_right()

        # print out observation
        # print("\rWriting Buffer: obs {}, new obs {}, action {}, reward {}, done {}".format(obs, new_obs, action, reward, done))

        # add the observation to the replay buffer
        replay_buffer.add(obs, new_obs, action, reward, done)

        # save the observations, for testing , remove later after testing
        buffer_storage.append([obs, new_obs, action, reward, done])
        if time_steps % 25 == 0:
            save_buffer = open("/home/charlotte/PycharmProjects/Baxter/td3algorithm/temp/buffer.pkl", "wb")
            pickle.dump(buffer_storage, save_buffer)
            save_buffer.close()
            print("buffer updated")

        # set the current observation to the new observation
        obs = new_obs
        time_steps += 1

        # if finished, reset the arm and get the reset state
        if done:
            env.reset(arm)
            if arm == "left":
                obs = env.left_state()
            else:
                obs = env.right_state()

        if time_steps % 100 == 0:
            print("\rPopulating Buffer {}/{}.".format(time_steps, observation_steps))
        sys.stdout.flush()
