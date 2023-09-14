import numpy as np
import math

def AngleDiff(endAngle, startAngle):
    deltaAngle = endAngle - startAngle
    abs_deltaAngle = np.fabs(deltaAngle)
    abs_compleAngle = 2 * np.pi - abs_deltaAngle
    if abs_compleAngle < abs_deltaAngle:
        diffAngle = -1 * np.sign(deltaAngle) * abs_compleAngle
    else:
        diffAngle = deltaAngle
    return diffAngle

def CalcDistance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def CalcCurvature(curr_p, prev_p, next_p):
    denominator = CalcDistance(prev_p, curr_p) * CalcDistance(curr_p, next_p) * CalcDistance(prev_p, next_p)
    if math.fabs(denominator) < 1e-6:
        return 0
    return 2.0 * ((curr_p[0] - prev_p[0]) * (next_p[1] - prev_p[1]) - (curr_p[1] - prev_p[1]) * (
            next_p[0] - prev_p[0])) / denominator

def IsShiftPoint(curr_p, prev_p, next_p):
    dot_product = (curr_p[0] - prev_p[0]) * (next_p[0] - curr_p[0]) + (curr_p[1] - prev_p[1]) * (
            next_p[1] - curr_p[1])
    norm_vector1 = CalcDistance(prev_p, curr_p)
    norm_vector2 = CalcDistance(curr_p, next_p)
    cos_theta = dot_product / (norm_vector1 * norm_vector2)
    if (cos_theta < 0):
        return True
    return False

# def ConvertQuaternionToYaw(quaternion):
#     quaternion /= np.linalg.norm(quaternion)
#
#     rotation_matrix = np.array([
#         [1 - 2 * (quaternion[2]**2 + quaternion[3]**2), 2 * (quaternion[1] * quaternion[2] - quaternion[0] * quaternion[3]), 2 * (quaternion[1] * quaternion[3] + quaternion[0] * quaternion[2])],
#         [2 * (quaternion[1] * quaternion[2] + quaternion[0] * quaternion[3]), 1 - 2 * (quaternion[1]**2 + quaternion[3]**2), 2 * (quaternion[2] * quaternion[3] - quaternion[0] * quaternion[1])],
#         [2 * (quaternion[1] * quaternion[3] - quaternion[0] * quaternion[2]), 2 * (quaternion[2] * quaternion[3] + quaternion[0] * quaternion[1]), 1 - 2 * (quaternion[1]**2 + quaternion[2]**2)]
#     ])
#
#     yaw_radians = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
#     return yaw_radians

# def ConvertQuaternionToYaw(quaternion):
#     x, y, z, w = quaternion
#     t0 = +2.0 * (w * z + x * y)
#     t1 = +1.0 - 2.0 * (y * y + z * z)
#     yaw = math.atan2(t0, t1)
#     return yaw
#
# def ConvertYawToQuaternion(yaw):
#     quaternion = [0.0, 0.0, 0.0, 0.0]
#     quaternion[0] = 0.0
#     quaternion[1] = 0.0
#     quaternion[2] = math.sin(yaw / 2.0)
#     quaternion[3] = math.cos(yaw / 2.0)
#     return quaternion

import tf_transformations

def ConvertQuaternionToYaw(quaternion):
    _, _, yaw = tf_transformations.euler_from_quaternion(quaternion)
    return yaw
def ConvertYawToQuaternion(yaw):
    quat = tf_transformations.quaternion_from_euler(0, 0, yaw)
    return quat

def ExtractRectangularContourPoints(center_pose, contour_shape):
    x, y, theta = center_pose[0], center_pose[1], center_pose[2]
    length, width = contour_shape[0], contour_shape[1]
    half_width = width / 2
    half_length = length / 2

    x_bottom_left = x - half_length* math.cos(theta) - half_width * math.sin(theta)
    y_bottom_left = y - half_length * math.sin(theta) + half_width * math.cos(theta)
    
    x_top_left = x + half_length * math.cos(theta) - half_width * math.sin(theta)
    y_top_left = y + half_length * math.sin(theta) + half_width * math.cos(theta)

    x_top_right = x + half_length * math.cos(theta) + half_width * math.sin(theta)
    y_top_right = y + half_length * math.sin(theta) - half_width * math.cos(theta)

    x_bottom_right = x - half_length * math.cos(theta) + half_width * math.sin(theta)
    y_bottom_right = y - half_length * math.sin(theta) - half_width * math.cos(theta)

    return [(x_top_left, y_top_left), (x_top_right, y_top_right), (x_bottom_right, y_bottom_right), (x_bottom_left, y_bottom_left), (x_top_left, y_top_left)]

# def RecoverAllData(dir):
#     def load_array_from_text(file_name):
#         loaded_array = np.loadtxt(file_name, delimiter=',')
#         return loaded_array
#
#     x0 = load_array_from_text("../log_1/x0.txt")
#     xF = load_array_from_text("../log_1/xF.txt")
#     u0 = load_array_from_text("../log_1/u0.txt")
#     ref_path = load_array_from_text("../log_1/ref_path.txt")
#     ref_input = load_array_from_text("../log_1/ref_input.txt")
#     raw_ref_path = load_array_from_text("../log_1/raw_ref_path.txt")
#     return x0, xF, u0, ref_path, ref_input, raw_ref_path

import os
def RecoverAllData(dirname):
    def load_array_from_text(file_name):
        loaded_array = np.loadtxt(file_name, delimiter=',')
        return loaded_array

    x0 = load_array_from_text(os.path.join(dirname, "x0.txt"))
    xF = load_array_from_text(os.path.join(dirname, "xF.txt"))
    u0 = load_array_from_text(os.path.join(dirname, "u0.txt"))
    ref_path = load_array_from_text(os.path.join(dirname, "ref_path.txt"))
    ref_input = load_array_from_text(os.path.join(dirname, "ref_input.txt"))
    raw_ref_path = load_array_from_text(os.path.join(dirname, "raw_ref_path.txt"))

    return x0, xF, u0, ref_path, ref_input, raw_ref_path
"""
Cubic spline planner

Author: Atsushi Sakai(@Atsushi_twi)

"""
import math
import numpy as np
import bisect


class CubicSpline1D:
    """
    1D Cubic Spline class

    Parameters
    ----------
    x : list
        x coordinates for data points. This x coordinates must be
        sorted
        in ascending order.
    y : list
        y coordinates for data points

    Examples
    --------
    You can interpolate 1D data points.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(5)
    >>> y = [1.7, -6, 5, 6.5, 0.0]
    >>> sp = CubicSpline1D(x, y)
    >>> xi = np.linspace(0.0, 5.0)
    >>> yi = [sp.calc_position(x) for x in xi]
    >>> plt.plot(x, y, "xb", label="Data points")
    >>> plt.plot(xi, yi , "r", label="Cubic spline interpolation")
    >>> plt.grid(True)
    >>> plt.legend()
    >>> plt.show()

    .. image:: cubic_spline_1d.png

    """

    def __init__(self, x, y):

        h = np.diff(x)
        if np.any(h < 0):
            raise ValueError("x coordinates must be sorted in ascending order")

        self.a, self.b, self.c, self.d = [], [], [], []
        self.x = x
        self.y = y
        self.nx = len(x)  # dimension of x

        # calc coefficient a
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h, self.a)
        self.c = np.linalg.solve(A, B)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            d = (self.c[i + 1] - self.c[i]) / (3.0 * h[i])
            b = 1.0 / h[i] * (self.a[i + 1] - self.a[i]) \
                - h[i] / 3.0 * (2.0 * self.c[i] + self.c[i + 1])
            self.d.append(d)
            self.b.append(b)

    def calc_position(self, x):
        """
        Calc `y` position for given `x`.

        if `x` is outside the data point's `x` range, return None.

        Returns
        -------
        y : float
            y position for given x.
        """
        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        position = self.a[i] + self.b[i] * dx + \
            self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return position

    def calc_first_derivative(self, x):
        """
        Calc first derivative at given x.

        if x is outside the input x, return None

        Returns
        -------
        dy : float
            first derivative for given x.
        """

        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        dy = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return dy

    def calc_second_derivative(self, x):
        """
        Calc second derivative at given x.

        if x is outside the input x, return None

        Returns
        -------
        ddy : float
            second derivative for given x.
        """

        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        ddy = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return ddy

    def __search_index(self, x):
        """
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        """
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        return A

    def __calc_B(self, h, a):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (a[i + 2] - a[i + 1]) / h[i + 1]\
                - 3.0 * (a[i + 1] - a[i]) / h[i]
        return B


class CubicSpline2D:
    """
    Cubic CubicSpline2D class

    Parameters
    ----------
    x : list
        x coordinates for data points.
    y : list
        y coordinates for data points.

    Examples
    --------
    You can interpolate a 2D data points.

    >>> import matplotlib.pyplot as plt
    >>> x = [-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0]
    >>> y = [0.7, -6, 5, 6.5, 0.0, 5.0, -2.0]
    >>> ds = 0.1  # [m] distance of each interpolated points
    >>> sp = CubicSpline2D(x, y)
    >>> s = np.arange(0, sp.s[-1], ds)
    >>> rx, ry, ryaw, rk = [], [], [], []
    >>> for i_s in s:
    ...     ix, iy = sp.calc_position(i_s)
    ...     rx.append(ix)
    ...     ry.append(iy)
    ...     ryaw.append(sp.calc_yaw(i_s))
    ...     rk.append(sp.calc_curvature(i_s))
    >>> plt.subplots(1)
    >>> plt.plot(x, y, "xb", label="Data points")
    >>> plt.plot(rx, ry, "-r", label="Cubic spline path")
    >>> plt.grid(True)
    >>> plt.axis("equal")
    >>> plt.xlabel("x[m]")
    >>> plt.ylabel("y[m]")
    >>> plt.legend()
    >>> plt.show()

    .. image:: cubic_spline_2d_path.png

    >>> plt.subplots(1)
    >>> plt.plot(s, [np.rad2deg(iyaw) for iyaw in ryaw], "-r", label="yaw")
    >>> plt.grid(True)
    >>> plt.legend()
    >>> plt.xlabel("line length[m]")
    >>> plt.ylabel("yaw angle[deg]")

    .. image:: cubic_spline_2d_yaw.png

    >>> plt.subplots(1)
    >>> plt.plot(s, rk, "-r", label="curvature")
    >>> plt.grid(True)
    >>> plt.legend()
    >>> plt.xlabel("line length[m]")
    >>> plt.ylabel("curvature [1/m]")

    .. image:: cubic_spline_2d_curvature.png
    """

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = CubicSpline1D(self.s, x)
        self.sy = CubicSpline1D(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        """
        calc position

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        x : float
            x position for given s.
        y : float
            y position for given s.
        """
        x = self.sx.calc_position(s)
        y = self.sy.calc_position(s)

        return x, y

    def calc_curvature(self, s):
        """
        calc curvature

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        k : float
            curvature for given s.
        """
        dx = self.sx.calc_first_derivative(s)
        ddx = self.sx.calc_second_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
        return k

    def calc_yaw(self, s):
        """
        calc yaw

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        yaw : float
            yaw angle (tangent vector) for given s.
        """
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        yaw = math.atan2(dy, dx)
        return yaw


def calc_spline_course(x, y, ds=0.1):
    sp = CubicSpline2D(x, y)
    s = list(np.arange(0, sp.s[-1], ds))

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s


def main_1d():
    print("CubicSpline1D test")
    import matplotlib.pyplot as plt
    x = np.arange(5)
    y = [1.7, -6, 5, 6.5, 0.0]
    sp = CubicSpline1D(x, y)
    xi = np.linspace(0.0, 5.0)

    plt.plot(x, y, "xb", label="Data points")
    plt.plot(xi, [sp.calc_position(x) for x in xi], "r",
             label="Cubic spline interpolation")
    plt.grid(True)
    plt.legend()
    plt.show()


def main_2d():  # pragma: no cover
    print("CubicSpline1D 2D test")
    import matplotlib.pyplot as plt
    ref_path[0, 0] = x0
    ref_path[1, 0] = y0
    ref_path[0, -1] = x1
    ref_path[1, -1] = y1

    x = [-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0]
    y = [0.7, -6, 5, 6.5, 0.0, 5.0, -2.0]
    ds = 0.1  # [m] distance of each interpolated points

    sp = CubicSpline2D(x, y)
    s = np.arange(0, sp.s[-1], ds)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    plt.subplots(1)
    plt.plot(x, y, "xb", label="Data points")
    plt.plot(rx, ry, "-r", label="Cubic spline path")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()

    plt.subplots(1)
    plt.plot(s, [np.rad2deg(iyaw) for iyaw in ryaw], "-r", label="yaw")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("yaw angle[deg]")

    plt.subplots(1)
    plt.plot(s, rk, "-r", label="curvature")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("curvature [1/m]")

    plt.show()



if __name__ == '__main__':
    quat1 = [0.0, 0.0,0.53729960834682389, 0.84339144581288561]
    quat2 = [0.0, 0.0, 0.53740123327716438,0.84332669498373092]
    yaw1 = ConvertQuaternionToYaw(quat1)
    yaw1_deg = np.rad2deg(yaw1)
    yaw2 = ConvertQuaternionToYaw(quat2)
    yaw2_deg = np.rad2deg(yaw2)
    angdiff = AngleDiff(yaw2, yaw1)
    angdiff_deg = np.rad2deg(angdiff)
    a = 0

