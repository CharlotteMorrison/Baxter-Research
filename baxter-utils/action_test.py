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
    # get an incremental update to the joint positions.
    right_pos = action.right_incr_action()
    left_pos = action.left_incr_action()

    # move the joints to the new positions.
    limb_right.move_to_joint_positions(right_pos)
    limb_left.move_to_joint_positions(left_pos)

    # update the current joint angles (could do this by using the positions
    # concern, there may be variability with movement, especially if the robot is holding
    # a weighty object, might need to reconsider to speed up processing.
    # TODO determine if getting updates significantly slows and if there is a difference between calculated angles
    action.action_update(limb_right.joint_angles(), limb_left.joint_angles())

    # get the endpoint pose for reward calculation
    right_reward.update_gripper(limb_right.endpoint_pose())
    left_reward.update_gripper(limb_left.endpoint_pose())

    # get the rewards and status
    right_distance, right_done = right_reward.euclidean_distance()
    left_distance, left_done = left_reward.euclidean_distance()

    print "Reward for iteration " + str(i)
    print right_reward.euclidean_distance()
    print left_reward.euclidean_distance()

    if left_done or right_done:
        # block_sample.delete_gazebo_models()
        quit()

block_sample.delete_gazebo_models()
quit()
