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
