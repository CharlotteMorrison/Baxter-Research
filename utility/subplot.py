from matplotlib import pyplot as plt
from IPython.display import clear_output


def subplot(R, P, Q, S):
    r = list(zip(*R))
    p = list(zip(*P))
    q = list(zip(*Q))
    s = list(zip(*S))
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