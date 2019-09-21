import unittest
from utility.action import Action


class TestAction(unittest.TestCase):
    def setUp(self):
        self.right = {'right_s0': 0.00, 'right_s1': 0.00, 'right_w0': 0.00,
                      'right_w1': 0.00, 'right_w2': 0.00, 'right_e0': 0.00, 'right_e1': 0.00}
        self.left = {'left_s0': 0.00, 'left_s1': 0.00, 'left_w0': 0.00,
                     'left_w1': 0.00, 'left_w2': 0.00, 'left_e0': 0.00, 'left_e1': 0.00}
        self.action = Action(self.right, self.left)
        self.update = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    def test_right_param_action(self):
        update_right = {'right_s0': 1.00, 'right_s1': 1.00, 'right_w0': 1.00,
                        'right_w1': 1.00, 'right_w2': 1.00, 'right_e0': 1.00, 'right_e1': 1.00}

        test_right = self.action.right_param_action(self.update)
        self.assertEqual(update_right, test_right)

    def test_left_param_action(self):
        update_left = {'left_s0': 1.00, 'left_s1': 1.00, 'left_w0': 1.00,
                       'left_w1': 1.00, 'left_w2': 1.00, 'left_e0': 1.00, 'left_e1': 1.00}

        test_left = self.action.left_param_action(self.update)
        self.assertEqual(update_left, test_left)


if __name__ == '__main__':
    unittest.main()
