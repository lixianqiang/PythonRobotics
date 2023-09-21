import numpy as np
import math


# import tf_transformations


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


def ConvertQuaternionToYaw(quaternion):
    x, y, z, w = quaternion
    t0 = +2.0 * (w * z + x * y)
    t1 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t0, t1)
    return yaw


def ConvertYawToQuaternion(yaw):
    quaternion = [0.0, 0.0, 0.0, 0.0]
    quaternion[0] = 0.0
    quaternion[1] = 0.0
    quaternion[2] = math.sin(yaw / 2.0)
    quaternion[3] = math.cos(yaw / 2.0)
    return quaternion


# def ConvertQuaternionToYaw(quaternion):
#     _, _, yaw = tf_transformations.euler_from_quaternion(quaternion)
#     return yaw


# def ConvertYawToQuaternion(yaw):
#     quat = tf_transformations.quaternion_from_euler(0, 0, yaw)
#     return quat


def ExtractRectangularContourPoints(center_pose, contour_shape):
    x, y, theta = center_pose[0], center_pose[1], center_pose[2]
    length, width = contour_shape[0], contour_shape[1]
    half_width = width / 2
    half_length = length / 2

    x_bottom_left = x - half_length * math.cos(theta) - half_width * math.sin(theta)
    y_bottom_left = y - half_length * math.sin(theta) + half_width * math.cos(theta)

    x_top_left = x + half_length * math.cos(theta) - half_width * math.sin(theta)
    y_top_left = y + half_length * math.sin(theta) + half_width * math.cos(theta)

    x_top_right = x + half_length * math.cos(theta) + half_width * math.sin(theta)
    y_top_right = y + half_length * math.sin(theta) - half_width * math.cos(theta)

    x_bottom_right = x - half_length * math.cos(theta) + half_width * math.sin(theta)
    y_bottom_right = y - half_length * math.sin(theta) - half_width * math.cos(theta)

    return [(x_top_left, y_top_left), (x_top_right, y_top_right), (x_bottom_right, y_bottom_right),
            (x_bottom_left, y_bottom_left), (x_top_left, y_top_left)]


def list2coord(obj):
    tf_obj = []
    for xy in zip(obj[0], obj[1]):
        tf_obj.append(xy)
    return tf_obj


if __name__ == '__main__':
    quat1 = [0.0, 0.0, 0.53729960834682389, 0.84339144581288561]
    quat2 = [0.0, 0.0, 0.53740123327716438, 0.84332669498373092]
    yaw1 = ConvertQuaternionToYaw(quat1)
    yaw1_deg = np.rad2deg(yaw1)
    yaw2 = ConvertQuaternionToYaw(quat2)
    yaw2_deg = np.rad2deg(yaw2)
    angdiff = AngleDiff(yaw2, yaw1)
    angdiff_deg = np.rad2deg(angdiff)
    a = 0
