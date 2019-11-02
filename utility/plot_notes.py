import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np


if __name__ == '__main__':
    fig = plt.figure()
    ax = plt.axes()
    x = np.linspace(0, 10, 1000)
    ax.plot(x, np.sin(x))
    plt.show()
