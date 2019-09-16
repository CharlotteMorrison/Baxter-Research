from matplotlib import pyplot as plt
from IPython.display import clear_output


def reward_subplot(avg_reward, episode_reward, total_timesteps):
    r = list(zip(*avg_reward))
    p = list(zip(*episode_reward))
    s = list(zip(*total_timesteps))
    clear_output(wait=True)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

    ax[0, 0].plot(list(r[1]), list(r[0]), 'r')  # row=0, col=0
    ax[1, 0].plot(list(p[1]), list(p[0]), 'b')  # row=1, col=0
    ax[1, 1].plot(list(s[1]), list(s[0]), 'k')  # row=1, col=1
    ax[0, 0].title.set_text('Average Reward')
    ax[1, 0].title.set_text('Episode Reward')
    ax[1, 1].title.set_text('Max steps')
    plt.show()


def subplot(reward, policy_loss, q_loss, max_steps):
    r = list(zip(*reward))
    p = list(zip(*policy_loss))
    q = list(zip(*q_loss))
    s = list(zip(*max_steps))
    clear_output(wait=True)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

    ax[0, 0].plot(list(r[1]), list(r[0]), 'r')  # row=0, col=0
    ax[1, 0].plot(list(p[1]), list(p[0]), 'b')  # row=1, col=0
    ax[0, 1].plot(list(q[1]), list(q[0]), 'g')  # row=0, col=1
    ax[1, 1].plot(list(s[1]), list(s[0]), 'k')  # row=1, col=1
    ax[0, 0].title.set_text('Reward')
    ax[1, 0].title.set_text('Policy loss')
    ax[0, 1].title.set_text('Q loss')
    ax[1, 1].title.set_text('Max steps')
    plt.show()
