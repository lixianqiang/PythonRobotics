import unittest
import numpy as np
from OBCA import obca
from pypoman import compute_polytope_halfspaces


class Test_GetHyperPlaneParam_Function(unittest.TestCase):

    def test_outside_of_obstacle(self):
        obj = [(-1, 2), (0, 3), (3, 2), (1, 0), (-1, 2)]
        A, b = obca.GetHyperPlaneParam(obj)
        x = np.array([[-1], [1]])
        result = A @ x < b
        desired_result = np.array([[True], [True], [True], [False]])
        self.assertTrue(np.all(result == desired_result))


    def test_inside_of_obstacle(self):
        obj = [(-1, 2), (0, 3), (3, 2), (1, 0), (-1, 2)]
        A, b = obca.GetHyperPlaneParam(obj)
        x = np.array([[1], [2]])
        result = A @ x < b
        desired_result = np.array([[True], [True], [True], [True]])
        self.assertTrue(np.all(result == desired_result))

    def test_xxx(self):
        # obj = [[20, 2.3], [11.3, 2.3], [11.3, 0], [20, 0], [20, 2.3]]
        # obj = [[6.0, 2.3], [0, 2.3], [0, 0], [6.0, 0], [6.0, 2.3]]
        obj = [(-1, 2), (0, 3), (3, 2), (1, 0), (-1, 2)]
        obj = [(-15, 0), (-15, 5), (-1.5, 5), (-1.5, 0), (-15, 0)]
        obstacles = [obj]
        for i in range(len(obstacles)):
            obs = obstacles[i][:-1]
            A_i, b_i = compute_polytope_halfspaces(obs)
            b_i = b_i.reshape(-1, 1)
            A, b = obca.GetHyperPlaneParam(obstacles[i])
            for j in range(len(A)):
                a, a_i = A[1, :], A_i[1, :]
                dot_product = np.dot(a, a_i)
                norm_A = np.linalg.norm(a)
                norm_A_i = np.linalg.norm(a_i)
                cos_angle = dot_product / (norm_A * norm_A_i)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)


if __name__ == '__main__':
    unittest.main()
