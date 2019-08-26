"""
Calculate the reward using the gripper location and the object location
"""
import numpy as np


class Reward (object):

    def __init__(self, gripper, goal):
        """
        The reward class takes in the gripper location and the goal location.
        These values are utilized to calculate a reward.
        :param gripper (dict): position and orientation as named tuples in a dict
                        pose = {'position': (x, y, z), 'orientation': (x, y, z, w)}
        :param goal (tuple): position in tuple (x, y, z)
        """
        self.gripper_pos = np.asarray(gripper.get("position"))
        # gripper orientation is unused, added for later implementation
        self.gripper_orient = gripper.get("orientation")
        self.goal = np.asarray(goal)

    def update_gripper(self, gripper):
        self.gripper_pos = np.asarray(gripper.get("position"))

    def euclidean_distance(self):
        distance = np.linalg.norm(self.gripper_pos - self.goal)
        return distance
