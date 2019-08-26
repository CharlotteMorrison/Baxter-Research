"""
Script to test the action class.
"""

import rospy
import baxter_interface
from action import Action
import block_sample
from reward import Reward

# initialize ROS node.
rospy.init_node('action_test')

# create instances of baxter_interfaces's limb class
limb_right = baxter_interface.Limb('right')
limb_left = baxter_interface.Limb('left')

block_sample.load_gazebo_models()


# create a new action for both arms
action = Action(limb_right.joint_angles(), limb_left.joint_angles())
right_reward = Reward(limb_right.endpoint_pose(), (0.6725, 0.1265, 0.7825))  # param 2 is block location
left_reward = Reward(limb_left.endpoint_pose(), (0.6725, 0.1265, 0.7825))  # param 2 is block location

# set the arms to a neutral position
limb_right.move_to_joint_positions(action.right_home_position())
limb_left.move_to_joint_positions(action.left_home_position())

for i in range(100):
    right_pos = action.right_incr_action()
    left_pos = action.left_incr_action()

    limb_right.move_to_joint_positions(right_pos)
    limb_left.move_to_joint_positions(left_pos)

    action.action_update(limb_right.joint_angles(), limb_left.joint_angles())

    right_reward.update_gripper(limb_right.endpoint_pose())
    left_reward.update_gripper(limb_left.endpoint_pose())

    print "Reward for iteration " + str(i)
    print right_reward.euclidean_distance()
    print left_reward.euclidean_distance()

block_sample.delete_gazebo_models()
quit()
