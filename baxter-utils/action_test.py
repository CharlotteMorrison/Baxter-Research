"""
Script to test the action class.
"""

import rospy
import baxter_interface
from action import Action
import block_sample

# initialize ROS node.
rospy.init_node('action_test')

# create instances of baxter_interfaces's limb class
limb_right = baxter_interface.Limb('right')
limb_left = baxter_interface.Limb('left')

block_sample.load_gazebo_models()


# create a new action for both arms
action = Action(limb_right.joint_angles(), limb_left.joint_angles())

for i in range(100):
    right_pos = action.right_incr_action()
    left_pos = action.left_incr_action()

    limb_right.move_to_joint_positions(right_pos)
    limb_left.move_to_joint_positions(left_pos)

    action.action_update(limb_right.joint_angles(), limb_left.joint_angles())

block_sample.delete_gazebo_models()
quit()
