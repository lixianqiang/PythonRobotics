"""

OBCA path planning

author: LiXianQiang(@lixianqiang)
e-mail: lxq243808918@gmail.com

reference:
    X. Zhang, A. Liniger and F. Borrelli, "Optimization-Based Collision Avoidance," in IEEE Transactions on Control Systems Technology, vol. 29, no. 3, pp. 972-983, May 2021, doi: 10.1109/TCST.2019.2949540.
    X. Zhang, A. Liniger, A. Sakai and F. Borrelli, "Autonomous Parking Using Optimization-Based Collision Avoidance," 2018 IEEE Conference on Decision and Control (CDC), Miami, FL, USA, 2018, pp. 4327-4332, doi: 10.1109/CDC.2018.8619433.
"""
import sys
import pathlib
import casadi as ca
import numpy as np

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from HybridAStar.hybrid_a_star import *

show_animation = True


def TransferDataForHybridAStar(map, obstalces, XY_GRID_RESOLUTION):
    interval = XY_GRID_RESOLUTION

    def generate_edge(start, end, ds):
        num_steps = int(np.linalg.norm(np.array(end) - np.array(start)) / ds)
        t_values = np.linspace(0, 1, num_steps)

        interpolated_points = [(start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1])) for t in
                               t_values]
        return interpolated_points

    coordinates = []
    for i in range(1, len(map)):
        p1, p2 = map[i - 1], map[i]
        coordinates.extend(generate_edge(p1, p2, interval))
        coordinates.pop()

    def generate_obstacle(vertices, interval):
        x_coords, y_coords = zip(*vertices)
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        x_range = np.arange(min_x, max_x + interval, interval)
        y_range = np.arange(min_y, max_y + interval, interval)

        obstacle_points = []
        for x in x_range:
            for y in y_range:
                obstacle_points.append((x, y))

        return obstacle_points

    for i in range(len(obstalces)):
        coordinates.extend(generate_obstacle(obstalces[i], interval))

    ox, oy = [], []
    for i in range(len(coordinates)):
        ox.append(coordinates[i][0])
        oy.append(coordinates[i][1])
    return ox, oy


def GetReferenceFromHybirdAStar(path, ego, x0, dt):
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

    # 去除重复节点
    for i in range(len(path.x_list) - 1, 0, -1):
        if (path.x_list[i] == path.x_list[i - 1] and path.y_list[i] == path.y_list[i - 1]):
            path.x_list.pop(i)
            path.y_list.pop(i)
            path.yaw_list.pop(i)
    # 去除距离过近的节点
    for i in range(len(path.x_list) - 1, 0, -1):
        p1 = [path.x_list[i - 1], path.y_list[i - 1]]
        p2 = [path.x_list[i], path.y_list[i]]
        if (CalcDistance(p1, p2) < 0.01):
            path.x_list.pop(i)
            path.y_list.pop(i)
            path.yaw_list.pop(i)

    x_list = path.x_list
    y_list = path.y_list
    yaw_list = path.yaw_list

    forward_list = []
    shift_list = []
    vel_list = []
    acc_list = []
    steer_list = []

    if (math.fabs(AngleDiff(math.atan2((y_list[1] - y_list[0]), (x_list[1] - x_list[0])), x0[3])) < np.pi / 2):
        orientation = 1
        forward_list.append(orientation)
        shift_list.append(False)
        vel_list.append(x0[3])

    else:
        orientation = -1
        forward_list.append(orientation)
        shift_list.append(False)
        vel_list.append(x0[3])

    for i in range(1, len(x_list) - 1):
        p1 = [x_list[i - 1], y_list[i - 1]]
        p2 = [x_list[i], y_list[i]]
        p3 = [x_list[i + 1], y_list[i + 1]]

        if (IsShiftPoint(p2, p1, p3)):
            orientation = -1 * orientation
            forward_list.append(orientation)
            shift_list.append(True)
            vel_list.append(0)
            acc_list.append(0)
            steer_list.append(steer_list[-1])
        else:
            ds, k = CalcDistance(p1, p2), CalcCurvature(p2, p1, p3)
            forward_list.append(orientation)
            shift_list.append(False)
            vel_list.append(orientation * ds / dt)
            acc_list.append((vel_list[i] - vel_list[i - 1]) / dt)
            steer_list.append(orientation * math.atan2(ego["wheel_base"] * k, 1))
    forward_list.append(forward_list[-1])
    shift_list.append(False)
    vel_list.append(0)
    acc_list.append(0)
    steer_list.append(steer_list[-1])

    ref_path = np.zeros((4, len(x_list)))
    for i in range(len(x_list)):
        ref_path[0, i] = x_list[i]
        ref_path[1, i] = y_list[i]
        ref_path[2, i] = yaw_list[i]
        ref_path[3, i] = vel_list[i]

    ref_input = np.zeros((2, len(acc_list)))
    for i in range(len(acc_list)):
        ref_input[0, i] = acc_list[i]
        ref_input[1, i] = steer_list[i]
    return ref_path, ref_input


def GetObstacleInfo(obstacle_list):
    num = len(obstacle_list)
    idx = [(i, i + 4) for i in range(0, num * 4, 4)]

    vlist = [len(obstacle_list[i]) - 1 for i in range(len(obstacle_list))]
    vnum = sum(vlist)

    cnt, vidx = 0, []
    for i in range(len(vlist)):
        vidx.append((cnt, cnt + vlist[i]))
        cnt += vlist[i]

    A_all = np.zeros((vnum, 2))
    b_all = np.zeros((vnum, 1))
    for i in range(num):
        A, b = GetHyperPlaneParam(obstacle_list[i])
        A_all[vidx[i][0]:vidx[i][1], :] = A
        b_all[vidx[i][0]:vidx[i][1], :] = b

    obs_info = {"num": num,
                "idx": idx,
                "vnum": vnum,
                "vlist": vlist,
                "vidx": vidx,
                "A": A_all,
                'b': b_all}
    return obs_info


# def GetHyperPlaneParam(obj):
#     # 论文假定障碍物的轮廓点以顺时针方式存放，并且满足Ax<b，其物理含义，障碍物被定义为由轮廓点顺时针所连接成的向量（超平面）的右侧。那么其法向量应该正交于超平面并指向左侧
#     # 即法向量是超平面以逆时针方向旋转90度得到，对于这种情况存在一种这样的关系 a = [vx, vy]， b = 【-vy, vx】所以这里与论文提供的源代码是等价
#     A = np.zeros((len(obj) - 1, 2))
#     b = np.zeros((len(obj) - 1, 1))
#     for i in range(len(obj) - 1):
#         v1 = obj[i]
#         v2 = obj[i + 1]
#         dx = v2[0] - v1[0]
#         dy = v2[1] - v1[1]
#         A[i, :] = np.array([[-dy, dx]])
#         b[i] = -v1[0] * dy + v1[1] * dx
#     return A, b

# tt=np.array([[15],[0]]);yy=np.array([[1.5],[5]]);cc=np.array([[1.5],[0]]);dd=np.array([[15],[5]])
def GetHyperPlaneParam(obj):
    A = np.zeros((len(obj) - 1, 2))
    b = np.zeros((len(obj) - 1, 1))
    for i in range(len(obj) - 1):
        v1 = obj[i]
        v2 = obj[i + 1]
        if v1[0] == v2[0]:
            if v2[1] < v1[1]:
                A_tmp = np.array([[1, 0]])
                b_tmp = v1[0]
            else:
                A_tmp = np.array([[-1, 0]])
                b_tmp = -v1[0]
        elif v1[1] == v2[1]:
            if v1[0] < v2[0]:
                A_tmp = np.array([[0, 1]])
                b_tmp = v1[1]
            else:
                A_tmp = [0, -1]
                b_tmp = -v1[1]
        else:
            ab = np.linalg.solve(np.array([[v1[0], 1], [v2[0], 1]]), np.array([[v1[1]], [v2[1]]]))
            ai = ab[0,0]
            bi = ab[1,0]
            if v1[0] < v2[0]:
                A_tmp = np.array([[-ai, 1]])
                b_tmp = bi
            else:
                A_tmp = np.array([[ai, -1]])
                b_tmp = -bi
        A[i, :] = A_tmp
        b[i] = b_tmp
    return A, b


def GetInitialDualVariable(ref_path, obstacles):
    obs_info = GetObstacleInfo(obstacles)
    N = ref_path.shape[1]
    X = ref_path[:3, :]

    # 定义变量
    opti = ca.Opti()
    L = opti.variable(obs_info["vnum"], N)
    M = opti.variable(obs_info['num'] * 4, N)
    D = opti.variable(obs_info['num'], N)

    # 目标函数
    objector_function = 0
    for k in range(N):
        for m in range(obs_info['num']):
            vidx, A_all = obs_info['vidx'][m], obs_info['A']
            l = ca.MX(L[vidx[0]:vidx[1], k])
            A = ca.MX(A_all[vidx[0]:vidx[1], :])
            norm2_square=ca.sumsqr(A.T @ l)
            objector_function = objector_function + D[m, k] + 0.5*norm2_square

    opti.minimize(objector_function)

    # 车辆边界描述
    g = ca.MX(np.array([[ego["length"] / 2], [ego["width"] / 2], [ego["length"] / 2], [ego["width"] / 2]]))
    G = ca.MX(np.array([[1, 0], [0, -1], [-1, 0], [0, 1]]))
    # 障碍物约束
    for k in range(N):
        Rot = ca.MX(np.array([[ca.cos(X[2, k]), -ca.sin(X[2, k])],
                              [ca.sin(X[2, k]), ca.cos(X[2, k])]]))
        T = ca.MX(np.array([[X[0, k] + ca.cos(X[2, k]) * ego["rear2center"]],
                            [X[1, k] + ca.sin(X[2, k]) * ego["rear2center"]]]))
        for m in range(obs_info['num']):
            idx, vidx, A_all, b_all = obs_info['idx'][m], obs_info['vidx'][m], obs_info['A'], obs_info['b']
            mu = M[idx[0]:idx[1], k]
            l = L[vidx[0]:vidx[1], k]
            A = ca.MX(A_all[vidx[0]:vidx[1], :])
            b = ca.MX(b_all[vidx[0]:vidx[1]])
            opti.subject_to(-g.T @ mu + (A @ T - b).T @ l + D[m, k] == 0)  # -g'*mu + (A*t - b)*lambda +dm==0
            opti.subject_to(G.T @ mu + Rot.T @ A.T @ l == 0)  # G'*mu + R'*A*lambda = 0
            opti.subject_to(D[m, k] < 0)
            opti.subject_to(l >= 0)
            opti.subject_to(mu >= 0)

    # 设置初始值
    opti.set_initial(L, np.zeros((obs_info['vnum'], N)))
    opti.set_initial(M, np.zeros((obs_info['num'] * 4, N)))
    opti.set_initial(D, -1*np.ones((obs_info['num'], N)))

    # 设置求解器
    options = {'ipopt.max_iter': 100, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
               'ipopt.acceptable_obj_change_tol': 1e-6}
    opti.solver('ipopt', options)
    sol = opti.solve()
    ref_L = sol.value(L)
    ref_M = sol.value(M)

    return ref_L, ref_M

def planning(x0, xF, u0, ego, XYbounds, obstacles, ref_path, ref_input, dt):
    dmin = 0.5
    obs_info = GetObstacleInfo(obstacles)
    N = ref_path.shape[1]
    # 定义变量
    opti = ca.Opti()
    X = opti.variable(4, N)
    U = opti.variable(2, N - 1)
    L = opti.variable(obs_info["vnum"], N)
    M = opti.variable(obs_info['num'] * 4, N)
    # 权重矩阵
    R = ca.diag([0.5, 0.5])
    Rd = ca.diag([0.1, 0.1])
    Q = ca.diag([0.0, 0.0, 0.0])
    # 目标函数
    object_function = 0
    for k in range(N - 1):
        object_function += ca.mtimes([U[:, k].T, R, U[:, k]])
    for k in range(1, N - 1):
        object_function += ca.mtimes([((U[:, k] - U[:, k - 1]) / dt).T, Rd, (U[:, k] - U[:, k - 1]) / dt])
    object_function += ca.mtimes([((U[:, 0] - u0) / dt).T, Rd, (U[:, 0] - u0) / dt])
    for k in range(N):
        object_function += ca.mtimes([(X[:3, k] - ref_path[:3, k]).T, Q, X[:3, k] - ref_path[:3, k]])
    opti.minimize(object_function)

    # 首末端状态约束
    opti.subject_to(X[:, 0] == x0)
    opti.subject_to(X[:, -1] == xF)

    # 自行车运动学模型
    func = lambda x, u, Ts: ca.vertcat(x[0] + Ts * x[3] * ca.cos(x[2]),
                                       x[1] + Ts * x[3] * ca.sin(x[2]),
                                       x[2] + Ts * x[3] * ca.tan(u[1]) / ego["wheel_base"],
                                       x[3] + Ts * u[0])
    # 运动学约束
    for k in range(N - 1):
        opti.subject_to(X[:, k + 1] == func(X[:, k], U[:, k], dt))

    # 状态约束
    for k in range(N):
        opti.subject_to(opti.bounded(XYbounds[0], X[0, k], XYbounds[1]))
        opti.subject_to(opti.bounded(XYbounds[2], X[1, k], XYbounds[3]))
        opti.subject_to(opti.bounded(-np.inf, X[2, k], np.inf))
        opti.subject_to(opti.bounded(-ego["max_vel"], X[3, k], ego["max_vel"]))

    # 输入约束
    for k in range(N - 1):
        opti.subject_to(opti.bounded(-ego["max_acc"], U[0, k], ego["max_acc"]))
        opti.subject_to(opti.bounded(-ego["max_steer"], U[1, k], ego["max_steer"]))
    for k in range(1, N - 1):
        opti.subject_to(opti.bounded(-ego["max_steer_rate"], (U[1, k] - U[1, k - 1]) / dt, ego["max_steer_rate"]))
    opti.subject_to(opti.bounded(-ego["max_steer_rate"], (U[1, 0] - u0[1]) / dt, ego["max_steer_rate"]))

    # 车辆边界描述
    g = ca.MX(np.array([[ego["length"] / 2], [ego["width"] / 2], [ego["length"] / 2], [ego["width"] / 2]]))
    G = ca.MX(np.array([[1, 0], [0, -1], [-1, 0], [0, 1]]))
    # 障碍物约束
    for k in range(N):
        Rot = ca.vertcat(ca.horzcat(ca.cos(X[2, k]), -ca.sin(X[2, k])),
                         ca.horzcat(ca.sin(X[2, k]), ca.cos(X[2, k])))
        T = ca.vertcat(X[0, k] + ca.cos(X[2, k]) * ego["rear2center"],
                       X[1, k] + ca.sin(X[2, k]) * ego["rear2center"])

        for m in range(obs_info['num']):
            idx, vidx, A_all, b_all = obs_info['idx'][m], obs_info['vidx'][m], obs_info['A'], obs_info['b']
            mu = M[idx[0]:idx[1], k]
            l = L[vidx[0]:vidx[1], k]
            A = ca.MX(A_all[vidx[0]:vidx[1], :])
            b = ca.MX(b_all[vidx[0]:vidx[1]])
            opti.subject_to(-g.T @ mu + (A @ T - b).T @ l > dmin)  # -g'*mu + (A*t - b)*lambda > dmin
            opti.subject_to(G.T @ mu + Rot.T @ A.T @ l == 0)  # G'*mu + R'*A*lambda = 0
            opti.subject_to(ca.norm_2(A.T @ l) <= 1)  # norm(A'*lambda) <= 1
            opti.subject_to(l >= 0)
            opti.subject_to(mu >= 0)

    # 设置初始值
    opti.set_initial(X, ref_path)
    opti.set_initial(U, ref_input)
    ref_L, ref_M = GetInitialDualVariable(ref_path, obstacles)
    opti.set_initial(L, ref_L)
    opti.set_initial(M, ref_M)

    # 设置求解器
    options = {'ipopt.max_iter': 2000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
               'ipopt.acceptable_obj_change_tol': 1e-6}
    opti.solver('ipopt', options)
    sol = opti.solve()
    trajectory = sol.value(X)
    input = sol.value(U)

    return trajectory, input


if __name__ == '__main__':
    dt = 0.6
    u0 = np.array([[0], [0]])
    x0 = np.array([[-10], [9.5], [0], [0]])
    xF = np.array([[0], [1.0], [np.pi / 2], [0]])
    ego = {"max_vel": 2,  # 这里的数据取值来源于HybridAStar的car.py
           "max_steer": 0.6,
           "max_acc": 0.4,
           "max_steer_rate": 0.6,
           "wheel_base": 3.0,
           "width": 2.0,
           "length": 3.3 + 1.0,  # LF + LB
           "rear_overhang": 1.0,
           "rear2center": (3.3 + 1.0) / 2 - 1.0  # length / 2 - rear_overhang
           }

    map = [(-15, 0), (-15, 15), (15, 15), (15, 0), (-15, 0)]
    obstacles = [[(-15, 0), (-15, 5), (-1.5, 5), (-1.5, 0), (-15, 0)],
                 [(1.5, 0), (1.5, 5), (15, 5), (15, 0), (1.5, 0)],
                 # [(-0, 10), (-0, 13), (2, 13), (2, 10), (-0, 10)],
                 ]
    x_coords, y_coords = zip(*map)
    XYBound = [min(x_coords), max(x_coords), min(y_coords), max(y_coords) - np.sqrt(1.0 ** 2 + 3.3 ** 2)]
    XY_GRID_RESOLUTION = 0.2  # [m]
    YAW_GRID_RESOLUTION = np.deg2rad(3.0)  # [rad]

    ox, oy = TransferDataForHybridAStar(map, obstacles, XY_GRID_RESOLUTION)
    hy_x0, hy_xF = x0.T.tolist()[0], xF.T.tolist()[0]
    if show_animation:
        plt.plot(ox, oy, ".k")
        rs.plot_arrow(hy_x0[0], hy_x0[1], hy_x0[2], fc='g')
        rs.plot_arrow(hy_xF[0], hy_xF[1], hy_xF[2])
        plt.grid(True)
        plt.axis("equal")
        plt.pause(0.0001)

    path = hybrid_a_star_planning(hy_x0[:3], hy_xF[:3], ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)
    path.x_list = path.x_list[::2]
    path.y_list = path.y_list[::2]
    path.yaw_list = path.yaw_list[::2]

    obstacles = [[(-15, 0), (-15, 5), (-1.5, 5), (-1.5, 0)],
                 [(1.5, 0), (1.5, 5), (15, 5)],
                 # [(-0, 10), (-0, 13), (2, 13), (2, 10), (-0, 10)],
                 ]



    ref_path, ref_input = GetReferenceFromHybirdAStar(path, ego, x0, dt)
    trajectory, input = planning(x0, xF, u0, ego, XYBound, obstacles, ref_path, ref_input, dt)

    x = trajectory[0]
    y = trajectory[1]
    yaw = trajectory[2]

    if show_animation:
        for t_x, t_y, t_yaw in zip(trajectory[0], trajectory[1], trajectory[2]):
            plt.cla()
            plt.plot(ox, oy, ".k")
            plt.plot(path.x_list, path.y_list, "-k", label="Hybrid A* path")
            plt.plot(x, y, "-r", label="OBCA path")
            plt.grid(True)
            plot_car(t_x, t_y, t_yaw)
            plt.axis("equal")
            plt.pause(0.05)
