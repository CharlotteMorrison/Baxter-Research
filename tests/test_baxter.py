import unittest
from utility.baxter_init import Baxter


class TestBaxter(unittest.TestCase):

    def setUp(self):
        self.baxter = Baxter()
        self.baxter.limb_right = {'right_s0': 0.00, 'right_s1': 0.00, 'right_w0': 0.00,
                                  'right_w1': 0.00, 'right_w2': 0.00, 'right_e0': 0.00, 'right_e1': 0.00}
        self.baxter.limb_left = {'left_s0': 0.00, 'left_s1': 0.00, 'left_w0': 0.00,
                                 'left_w1': 0.00, 'left_w2': 0.00, 'left_e0': 0.00, 'left_e1': 0.00}

    def test_step_left(self):
        self.fail()

    def test_step_right(self):
        self.fail()

    def test_random_step_left(self):
        self.fail()

    def test_random_step_right(self):
        self.fail()

    def test_observation_space(self):
        self.fail()

    def test_action_space(self):
        self.fail()

    def test_action_domain(self):
        self.fail()

    def test_reset(self):
        self.fail()
