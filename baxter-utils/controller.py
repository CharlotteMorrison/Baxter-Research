import argparse
import math
import random

import rospy

from std_msgs.msg import (
    UInt16,
)

import baxter_interface

from baxter_interface import CHECK_VERSION

class Controller(object):
    def __init__(self):
        """
        Moves the robots arms 
        """


