import unittest
import numpy as np
from pavlidis import pavlidis


class TestPavlidis(unittest.TestCase):
    def test_pixel(self):
        case = np.zeros((3, 4), np.uint8)
        case[1, 2] = True
        result = pavlidis(case, 1, 2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0, 0], 1)
        self.assertEqual(result[0, 1], 2)

    def test_edge(self):
        # Test a 2x2 square within a 2x2 grid so that it might run off
        # the edges
        # This checks turning too
        case = np.ones((2, 2), np.uint8)
        result = pavlidis(case, 1, 0)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0, 0], 1)
        self.assertEqual(result[0, 1], 0)
        self.assertEqual(result[1, 0], 0)
        self.assertEqual(result[1, 1], 0)
        self.assertEqual(result[2, 0], 0)
        self.assertEqual(result[2, 1], 1)
        self.assertEqual(result[3, 0], 1)
        self.assertEqual(result[3, 1], 1)

    def test_bool_cast(self):
        case = np.zeros((3, 3), bool)
        case[1, 1] = True
        result = pavlidis(case, 1, 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0, 0], 1)
        self.assertEqual(result[0, 1], 1)

    # The test names check the p1, p2 and p3 cases for the directions,
    # xn (going towards -x), yn, xp and yp
    #
    def test_p1xn(self):
        #
        #   2  0
        #   | /
        #   1
        case = np.zeros((4, 4), np.uint8)
        case[1, 1:3] = True
        case[2, 1] = True
        result = pavlidis(case, 1, 2)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0, 0], 1)
        self.assertEqual(result[0, 1], 2)
        self.assertEqual(result[1, 0], 2)
        self.assertEqual(result[1, 1], 1)

    def test_p2xn(self):
        #
        # 2
        # |
        # 1 - 0
        case = np.zeros((4, 4), np.uint8)
        case[2, 1:3] = True
        case[1, 1] = True
        result = pavlidis(case, 2, 2)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0, 0], 2)
        self.assertEqual(result[0, 1], 2)
        self.assertEqual(result[1, 0], 2)
        self.assertEqual(result[1, 1], 1)

    def test_p3xn(self):
        #
        #  1--2
        #   \
        #    0
        #
        case = np.zeros((4, 4), np.uint8)
        case[2, 2] = True
        case[1, 1:3] = True
        result = pavlidis(case, 2, 2)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0, 0], 2)
        self.assertEqual(result[0, 1], 2)
        self.assertEqual(result[1, 0], 1)
        self.assertEqual(result[1, 1], 1)

    def test_p1yn(self):
        #
        #    2--3
        #     \
        #       1
        #       |
        #       0
        case = np.zeros((4, 4), np.uint8)
        case[0:3, 2] = True
        case[0, 1] = True
        result = pavlidis(case, 2, 2)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[1, 0], 1)
        self.assertEqual(result[1, 1], 2)
        self.assertEqual(result[2, 0], 0)
        self.assertEqual(result[2, 1], 1)

    def test_p2yn(self):
        #
        #   2 -- 3
        #   |
        #   1
        #   |
        #   0
        case = np.zeros((4, 4), np.uint8)
        case[1:, 1] = True
        case[1,  2] = True
        result = pavlidis(case, 3, 1)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[1, 0], 2)
        self.assertEqual(result[1, 1], 1)
        self.assertEqual(result[2, 0], 1)
        self.assertEqual(result[2, 1], 1)

    def test_p3yn(self):
        #
        #     2
        #    /
        #   1
        #   |
        #   0
        case = np.zeros((4, 4), np.uint8)
        case[2:, 1] = True
        case[1,  2] = True
        result = pavlidis(case, 3, 1)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[1, 0], 2)
        self.assertEqual(result[1, 1], 1)
        self.assertEqual(result[2, 0], 1)
        self.assertEqual(result[2, 1], 2)

    def test_p1xp(self):
        #             3
        #            /|
        #      1 - 2  4
        #    /
        #  0
        case = np.zeros((4, 4), np.uint8)
        case[3, 0] = True
        case[2, 1:4] = True
        case[1, 3] = True
        result = pavlidis(case, 3, 0)
        self.assertEqual(len(result), 7)
        self.assertEqual(result[2, 0], 2)
        self.assertEqual(result[2, 1], 2)
        self.assertEqual(result[3, 0], 1)
        self.assertEqual(result[3, 1], 3)

    def test_p2xp(self):
        #
        #      1 - 2 - 3
        #    /
        #  0
        case = np.zeros((4, 4), np.uint8)
        case[3, 0] = True
        case[2, 1:4] = True
        result = pavlidis(case, 3, 0)
        self.assertEqual(len(result), 6)
        self.assertEqual(result[2, 0], 2)
        self.assertEqual(result[2, 1], 2)
        self.assertEqual(result[3, 0], 2)
        self.assertEqual(result[3, 1], 3)

    def test_p3xp(self):
        #
        #      1 - 2
        #    /      \
        #  0         3
        case = np.zeros((4, 4), np.uint8)
        case[3, 0] = True
        case[2, 1:3] = True
        case[3, 3] = True
        result = pavlidis(case, 3, 0)
        self.assertEqual(len(result), 6)
        self.assertEqual(result[2, 0], 2)
        self.assertEqual(result[2, 1], 2)
        self.assertEqual(result[3, 0], 3)
        self.assertEqual(result[3, 1], 3)

    def test_p1yp(self):
        #
        #      1 - 2
        #    /     |
        #  0       3
        #            \
        #          5--4
        case = np.zeros((4, 4), np.uint8)
        case[2, 0] = True
        case[1, 1] = True
        case[1, 2] = True
        case[2, 2] = True
        case[3, 2:] = True
        result = pavlidis(case, 2, 0)
        self.assertEqual(result[3, 0], 2)
        self.assertEqual(result[3, 1], 2)
        self.assertEqual(result[4, 0], 3)
        self.assertEqual(result[4, 1], 3)

    def test_p2yp(self):
        #
        #      1 - 2
        #    /     |
        #  0       3
        #          |
        #     5 -- 4
        case = np.zeros((4, 4), np.uint8)
        case[2, 0] = True
        case[1, 1] = True
        case[1, 2] = True
        case[2, 2] = True
        case[3, 1:3] = True
        result = pavlidis(case, 2, 0)
        self.assertEqual(result[3, 0], 2)
        self.assertEqual(result[3, 1], 2)
        self.assertEqual(result[4, 0], 3)
        self.assertEqual(result[4, 1], 2)

    def test_p3yp(self):
        #
        #      1 - 2
        #    /     |
        #  0       3
        #          |
        #     5 -- 4
        case = np.zeros((4, 4), np.uint8)
        case[2, 0] = True
        case[1, 1] = True
        case[1, 2] = True
        case[2, 2] = True
        case[3, 1] = True
        result = pavlidis(case, 2, 0)
        self.assertEqual(result[3, 0], 2)
        self.assertEqual(result[3, 1], 2)
        self.assertEqual(result[4, 0], 3)
        self.assertEqual(result[4, 1], 1)

if __name__ == '__main__':
    unittest.main()
