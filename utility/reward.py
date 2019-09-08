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
        # for use after moving the gripper, to provide a more current position.
        self.gripper_pos = np.asarray(gripper.get("position"))

    def euclidean_distance(self):
        # uses numpy euclidean distance, (more efficient than scipy implementation)
        distance = np.linalg.norm(self.gripper_pos - self.goal)
        # distance from object is subtracted from arm diameter
        # furthest possible distance-not feasible farthest possible
        distance = 2.04 - distance
        done = self.is_done(distance)
        return distance

    @classmethod
    def is_done(cls, distance):
        # needs a more robust solution for doneness, this is just temp for development
        if distance > 1.7:
            done = True
        else:
            done = False
        return done
