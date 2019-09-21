import unittest
from utility.reward import Reward
import numpy as np


class TestReward(unittest.TestCase):
    def setUp(self):
        self.reward = Reward({'position': (0.0, 0.0, 0.0), 'orientation': (1, 1, 1, 1)}, (0.6725, 0.1265, 0.7825))

    def test_euclidean_distance(self):
        distance = self.reward.euclidean_distance()
        self.assertAlmostEqual(distance, 1.000497835500089)

    def test_is_done(self):
        distance = 1.9
        result = self.reward.is_done(distance)
        self.assertTrue(result)

    def test_not_done(self):
        distance = 1.5
        result = Reward.is_done(distance)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
