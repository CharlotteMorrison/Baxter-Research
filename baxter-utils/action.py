import random
import numpy as np


def get_rand_position():
    """
        Gets a random position for each joint in an arm
        Currently set to random number
        Returns value in radians.
    """
    # Joint ranges from: ROS Robotics by Example by Carol Fairchild
    # angle measurements:                     degrees                             radians
    # ------------------------------------------------------------------------------------------------
    #                               min         max         range   |   min         max         range
    # S0: shoulder twist (roll)     -.97.494    +97.494     194.998 |   -1.7016     +1.7016     3.4033
    # S1: shoulder bend (pitch)     -123        +60         183     |   -2.147      +1.047      3.194
    # E0: elbow twist (roll)        -174.987    +174.987    349.979 |   -3.0541     +3.0541     6.1083
    # E1: elbow bend (pitch)        -2.864      +150        153     |   -.05        +2.618      2.67
    # W0: wrist twist (roll)        -175.25     +175.25     350.5   |   -3.059      +3.059      6.117
    # W1: wrist bend (pitch)        -90         +120        210     |   -1.5707     +2.094      3.6647
    # W2: wrist twist (roll)        -175.25     +175.25     350.5   |   -3.059      +3.059      6.117

    # get a random number from the range of degrees, rounded to in range.
    S0 = random.randrange(-97, 97, 1)
    S1 = random.randrange(-123, 60, 1)
    E0 = random.randrange(-174, 174, 1)
    E1 = random.randrange(-2, 150, 1)
    W0 = random.randrange(-175, 175, 1)
    W1 = random.randrange(-90, 120, 1)
    W2 = random.randrange(-175, 175, 1)

    # add arm values to list
    arm = [S0, S1, E0, E1, W0, W1, W2]

    # convert to list of radian values
    arm = np.radians(arm)

    # round values to two digits
    arm = [round(x, 2) for x in arm]

    return arm


class Action(object):

    def __init__(self):
        # default values for arm dictionary are the home position for the arms.
        # these values will not propagate to the robot, are provided for backup
        self.right_arm = {'right_s0': 0.08,
                          'right_s1': -1.00,
                          'right_w0': -0.67,
                          'right_w1': 1.03,
                          'right_w2': 0.50,
                          'right_e0': 1.18,
                          'right_e1': 1.94}

        self.left_arm = {'left_s0': -0.08,
                         'left_s1': -1.00,
                         'left_w0': 0.67,
                         'left_w1': 1.03,
                         'left_w2': -0.50,
                         'left_e0': -1.18,
                         'left_e1': 1.94}

    def right_action(self):
        action = get_rand_position()
        self.right_arm = {'right_s0': action[0],
                          'right_s1': action[1],
                          'right_w0': action[2],
                          'right_w1': action[3],
                          'right_w2': action[4],
                          'right_e0': action[5],
                          'right_e1': action[6]}
        return self.right_arm

    def left_action(self):
        action = get_rand_position()
        self.left_arm = {'left_s0': action[0],
                         'left_s1': action[1],
                         'left_w0': action[2],
                         'left_w1': action[3],
                         'left_w2': action[4],
                         'left_e0': action[5],
                         'left_e1': action[6]}
        return self.left_arm


# actions = Action()
# actions.left_action()
# actions.right_action()
