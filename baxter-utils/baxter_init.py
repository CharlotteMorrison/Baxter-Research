""" Creates an instance of baxter with both left and right limbs"""
import rospy
import baxter_interface
from action import Action
import block_sample
from reward import Reward


class Baxter(object):

    def __init__(self):
        # initialize baxter node in ROS
        rospy.init_node('baxter_node')

        # create instances of baxter_interfaces's limb class
        self.limb_right = baxter_interface.Limb('right')
        self.limb_left = baxter_interface.Limb('left')
        # load the gazebo model for simulation
        block_sample.load_gazebo_models()

        # create a new action for both arms
        self.action = Action(self.limb_right.joint_angles(), self.limb_left.joint_angles())
        self.neutral()

        # initialize the reward with goal location, param 2 is block location in sample
        # TODO use input from robot vision to get goal position
        self.right_reward = Reward(self.limb_right.endpoint_pose(), (0.6725, 0.1265, 0.7825))
        self.left_reward = Reward(self. limb_left.endpoint_pose(), (0.6725, 0.1265, 0.7825))

    def neutral(self):
        # set the arms to a neutral position
        self.limb_right.move_to_joint_positions(self.action.right_home_position())
        self.limb_left.move_to_joint_positions(self.action.left_home_position())

    def right_step(self):
        # get an incremental update to the joint positions.
        right_pos = self.action.right_incr_action()

        # move the joints to the new positions.
        self.limb_right.move_to_joint_positions(right_pos)

        # update current state of robot
        self.action.action_update(self.right_state(), self.left_state())

        # get the endpoint pose for reward calculation
        self.right_reward.update_gripper(self.limb_right.endpoint_pose())

        # get the rewards and status
        return right_pos, self.right_reward.euclidean_distance()

    def left_step(self):
        # get an incremental update to the joint positions.
        left_pos = self.action.left_incr_action()

        # move the joints to the new positions.
        self.limb_left.move_to_joint_positions(left_pos)

        # update current state of robot
        self.action.action_update(self.right_state(), self.left_state())

        # get the endpoint pose for reward of robot
        self.left_reward.update_gripper(self.limb_left.endpoint_pose())

        # return the reward and status
        return left_pos, self.left_reward.euclidean_distance()

    def right_state(self):
        # returns an array of the current joint angles
        return self.limb_right.joint_angles()

    def left_state(self):
        # returns an array of the current joint angles
        return self.limb_left.joint_angles()

    def close_env(self):
        block_sample.delete_gazebo_models()
        self.neutral()
        quit()
