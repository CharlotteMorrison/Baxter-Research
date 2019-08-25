"""
Script to test the action class.
"""

import rospy
import baxter_interface
from action import Action

# initialize ROS node.
rospy.init_node('random_arms')

# create instances of baxter_interfaces's limb class
limb_right = baxter_interface.Limb('right')
limb_left = baxter_interface.Limb('left')

# create random action to pass to limb

action = Action()

right_pos = action.right_action()
left_pos = action.left_action()

limb_right.move_to_joint_positions(right_pos)
limb_left.move_to_joint_positions(left_pos)

quit()
