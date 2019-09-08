""" Creates an instance of baxter with both left and right limbs"""
import numpy as np

import baxter_interface
import rospy

import block_sample
from action import Action
from reward import Reward
import random


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
        # TODO add a 2nd block, have each hand work towards their own goal.
        # TODO use input from robot vision to get goal position
        self.right_reward = Reward(self.limb_right.endpoint_pose(), (0.6725, 0.1265, 0.7825))
        self.left_reward = Reward(self. limb_left.endpoint_pose(), (0.6725, 0.1265, 0.7825))

    def neutral(self):
        # set the arms to a neutral position
        self.limb_right.move_to_joint_positions(self.action.right_home_position())
        self.limb_left.move_to_joint_positions(self.action.left_home_position())

    def step(self, side, action):
        if side == "right":
            start_pos = self.right_state()
            # get an incremental update to the joint positions.
            right_pos = self.action.right_param_action(action)

            # move the joints to the new positions.
            self.limb_right.move_to_joint_positions(right_pos)

            # update current state of robot
            self.action.action_update(self.right_state(), self.left_state())

            # get the endpoint pose for reward calculation
            self.right_reward.update_gripper(self.limb_right.endpoint_pose())
            reward = self.right_reward.euclidean_distance()
            done = self.right_reward.is_done(reward)
            # get the rewards and status
            return start_pos, right_pos, reward, done
        if side == "left":
            start_pos = self.left_state()
            # get an incremental update to the joint positions.
            left_pos = self.action.left_param_action(action)

            # move the joints to the new positions.
            self.limb_left.move_to_joint_positions(left_pos)

            # update current state of robot
            self.action.action_update(self.right_state(), self.left_state())

            # get the endpoint pose for reward of robot
            self.left_reward.update_gripper(self.limb_left.endpoint_pose())

            reward = self.left_reward.euclidean_distance()
            done = self.left_reward.is_done(reward)
            # return the reward and status
            print reward
            return start_pos, left_pos, reward, done
        else:
            return "Incorrect parameter:left or right"

    def random_step_left(self):
        init_arm_pos = self.left_state()
        item = 0
        new_arm_pos = init_arm_pos
        value = [-0.1, 0, 0.1]
        arm_update_value = []
        for i in range(7):
            arm_update_value.append(random.choice(value))
        for key, value in new_arm_pos.iteritems():
            # sets new arm position to initial position + a random value
            new_arm_pos[key] = init_arm_pos[key] + arm_update_value[item]
            item += 1

        self.limb_left.move_to_joint_positions(new_arm_pos)
        self.action.action_update(self.right_state(), self.left_state())
        self.left_reward.update_gripper(self.limb_left.endpoint_pose())

        reward = self.left_reward.euclidean_distance()
        done = self.left_reward.is_done(reward)
        return init_arm_pos, new_arm_pos, arm_update_value, reward, done

    def random_step_right(self):
        init_arm_pos = self.right_state()
        item = 0
        new_arm_pos = init_arm_pos
        value = [-0.1, 0, 0.1]
        arm_update_value = []
        for i in range(7):
            arm_update_value.append(random.choice(value))
        for key, value in new_arm_pos.iteritems():
            # sets new arm position to initial position + a random value
            new_arm_pos[key] = init_arm_pos[key] + arm_update_value[item]
            item += 1

        self.limb_right.move_to_joint_positions(new_arm_pos)
        self.action.action_update(self.right_state(), self.left_state())
        self.right_reward.update_gripper(self.limb_right.endpoint_pose())

        reward = self.right_reward.euclidean_distance()
        done = self.right_reward.is_done(reward)
        return init_arm_pos, new_arm_pos, arm_update_value, reward, done

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

    def observation_space(self):
        low = np.array([-1.7016, -2.147, -3.0541, -.05, -3.059, -1.5707, -3.059])
        high = np.array([1.7016, 1.047, 3.0541, 2.618, 3.059, 2.094, 3.059])
        return [low, high]
# total number of possible states

    def action_space(self):
        return [[-0.1, 0, 0.1], [-0.1, 0, 0.1], [-0.1, 0, 0.1], [-0.1, 0, 0.1], [-0.1, 0, 0.1], [-0.1, 0, 0.1],
                [-0.1, 0, 0.1]]
        # return [-0.1, 0, 0.1]

    def action_domain(self):
        low = -0.1
        # TODO may want to change this, add in a bigger value like .2
        none = 0.0
        high = 0.1
        return low, none, high

    def reset(self, arm):
        self.neutral()
        if arm == "right":
            state = self.right_state()
        elif arm == "left":
            state = self.left_state()
        else:
            print "non-valid arm parameter: enter 'left' or 'right'"
        state_list = list(state.values())
        return state_list

