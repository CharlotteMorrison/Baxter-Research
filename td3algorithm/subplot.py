from matplotlib import pyplot as plt
from IPython.display import clear_output


class Subplot(object):

    def __init__(self):
        self.reward_list = []
        plt.style.use('seaborn-whitegrid')
        self.fig = plt.figure()
        self.ax = plt.axes()

    def reward_subplot(self, current_reward, title):
        self.reward_list.append(current_reward)

        # TODO show plot every 100 steps
        if len(self.reward_list) % 2 == 0:

            clear_output(wait=True)
            plt.plot(self.reward_list, linestyle='solid')
            plt.title(title)
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.savefig('/home/charlotte/PycharmProjects/Baxter/graphs/' + title + '.png')
            plt.show()

