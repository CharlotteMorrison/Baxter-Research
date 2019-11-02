from td3algorithm.subplot import Subplot
import random


if __name__ == '__main__':

    test = Subplot()
    for i in range(300):
        test.reward_subplot(random.uniform(0.0, 2.0))
