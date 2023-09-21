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
from unit import *


def CalculateReferenceInput(ref_traj, ego):
    steer_list = []
    x_list, y_list, vel_list = ref_traj[0], ref_traj[1], ref_traj[3]
    orientation = np.sign(vel_list[1])
    for i in range(1, len(vel_list) - 1):
        if (math.fabs(vel_list[i]) < 1e-3 or vel_list[i - 1] * vel_list[i] < 0):
            orientation = -1 * orientation
            steer_list.append(steer_list[-1])
        else:
            p1 = [x_list[i - 1], y_list[i - 1]]
            p2 = [x_list[i], y_list[i]]
            p3 = [x_list[i + 1], y_list[i + 1]]
            k = CalcCurvature(p2, p1, p3)
            steer_list.append(orientation * math.atan2(ego["wheel_base"] * k, 1))
    steer_list.append(0.0)
    acc_list = []
    for j in range(1, len(vel_list) - 1):
        if (math.fabs(vel_list[j]) < 1e-3 or vel_list[j - 1] * vel_list[j] < 0):
            acc_list.append(0)
        else:
            acc_list.append(vel_list[j] - vel_list[j - 1])
    acc_list.append(0)
    ref_input = np.vstack((acc_list, steer_list))
    return ref_input


def GetObstacleInfo(obstacle_list):
    onum = len(obstacle_list)
    oidx_list = [(i, i + 4) for i in range(0, onum * 4, 4)]

    vnum_list = [len(obstacle_list[i]) - 1 for i in range(len(obstacle_list))]
    vsum = sum(vnum_list)

    cnt, vidx_list = 0, []
    for i in range(len(vnum_list)):
        vidx_list.append((cnt, cnt + vnum_list[i]))
        cnt += vnum_list[i]

    A_all = np.zeros((vsum, 2))
    b_all = np.zeros((vsum, 1))
    for i in range(onum):
        A, b = GetHyperPlaneParam(obstacle_list[i])
        A_all[vidx_list[i][0]:vidx_list[i][1], :] = A
        b_all[vidx_list[i][0]:vidx_list[i][1], :] = b

    obs_info = {"onum": onum,
                "oidx_list": oidx_list,
                "vsum": vsum,
                "vnum_list": vnum_list,
                "vidx_list": vidx_list,
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
            ai = ab[0, 0]
            bi = ab[1, 0]
            if v1[0] < v2[0]:
                A_tmp = np.array([[-ai, 1]])
                b_tmp = bi
            else:
                A_tmp = np.array([[ai, -1]])
                b_tmp = -bi
        A[i, :] = A_tmp
        b[i] = b_tmp
    return A, b


def GetInitialDualVariable(ref_traj, obstacles, ego):
    obs_info = GetObstacleInfo(obstacles)
    N = ref_traj.shape[1]
    X = ref_traj[:3, :]

    # 定义变量
    opti = ca.Opti()
    L = opti.variable(obs_info["vsum"], N)
    M = opti.variable(obs_info['onum'] * 4, N)
    D = opti.variable(obs_info['onum'], N)

    # 目标函数
    objector_function = 0
    for k in range(N):
        for m in range(obs_info['onum']):
            vidx_list, A_all = obs_info['vidx_list'][m], obs_info['A']
            l = ca.MX(L[vidx_list[0]:vidx_list[1], k])
            A = ca.MX(A_all[vidx_list[0]:vidx_list[1], :])
            norm2_square = ca.sumsqr(A.T @ l)
            objector_function = objector_function + D[m, k] + 0.5 * norm2_square

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
        for m in range(obs_info['onum']):
            oidx_list, vidx_list, A_all, b_all = obs_info['oidx_list'][m], obs_info['vidx_list'][m], obs_info['A'], \
                obs_info['b']
            mu = M[oidx_list[0]:oidx_list[1], k]
            l = L[vidx_list[0]:vidx_list[1], k]
            A = ca.MX(A_all[vidx_list[0]:vidx_list[1], :])
            b = ca.MX(b_all[vidx_list[0]:vidx_list[1]])
            opti.subject_to(-g.T @ mu + (A @ T - b).T @ l + D[m, k] == 0)  # -g'*mu + (A*t - b)*lambda +dm==0
            opti.subject_to(G.T @ mu + Rot.T @ A.T @ l == 0)  # G'*mu + R'*A*lambda = 0
            opti.subject_to(D[m, k] < 0)
            opti.subject_to(l >= 0)
            opti.subject_to(mu >= 0)

    # 设置初始值
    opti.set_initial(L, np.zeros((obs_info['vsum'], N)))
    opti.set_initial(M, np.zeros((obs_info['onum'] * 4, N)))
    opti.set_initial(D, -1 * np.ones((obs_info['onum'], N)))

    # 设置求解器
    options = {'ipopt.max_iter': 100, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
               'ipopt.acceptable_obj_change_tol': 1e-6}
    opti.solver('ipopt', options)
    sol = opti.solve()
    ref_L = sol.value(L)
    ref_M = sol.value(M)

    return ref_L, ref_M


def planning(x0, xF, u0, ego, map, obstacles, ref_traj, ref_input, dt=1.0):
    dmin = 0.5
    map.reverse()
    obstacles.extend(map2obstacle(map))
    obs_info = GetObstacleInfo(obstacles)
    N = ref_traj.shape[1]
    # 定义变量
    opti = ca.Opti()
    X = opti.variable(4, N)
    U = opti.variable(2, N - 1)
    L = opti.variable(obs_info["vsum"], N)
    M = opti.variable(obs_info['onum'] * 4, N)
    Ts = opti.variable(1, N)

    # 权重矩阵
    R = ca.diag([0.5, 0.5])
    Rd = ca.diag([0.1, 0.1])
    Q = ca.diag([0.1, 0.1, 0.1, 0.1])
    # 目标函数
    object_function = 0
    for k in range(N - 1):
        object_function += ca.mtimes([U[:, k].T, R, U[:, k]])
    for k in range(1, N - 1):
        object_function += ca.mtimes(
            [((U[:, k] - U[:, k - 1]) / (Ts[k] * dt)).T, Rd, (U[:, k] - U[:, k - 1]) / (Ts[k] * dt)])
    object_function += ca.mtimes([((U[:, 0] - u0) / (Ts[0] * dt)).T, Rd, (U[:, 0] - u0) / ((Ts[0] * dt))])
    for k in range(N):
        object_function += (0.5 * Ts[k] + Ts[k] * Ts[k])
    for k in range(N):
        object_function += ca.mtimes([(X[:, k] - ref_traj[:, k]).T, Q, X[:, k] - ref_traj[:, k]])
    opti.minimize(object_function)

    # 首末端状态约束
    opti.subject_to(X[:, 0] == x0)
    opti.subject_to(X[:, -1] == xF)

    # 自行车运动学模型
    func = lambda x, u, dT: ca.vertcat(x[0] + dT * x[3] * ca.cos(x[2]),
                                       x[1] + dT * x[3] * ca.sin(x[2]),
                                       x[2] + dT * x[3] * ca.tan(u[1]) / ego["wheel_base"],
                                       x[3] + dT * u[0])
    # 运动学约束
    for k in range(N - 1):
        opti.subject_to(X[:, k + 1] == func(X[:, k], U[:, k], Ts[k] * dt))

    # 状态约束
    for k in range(N):
        # opti.subject_to(opti.bounded(-np.inf, X[2, k], np.inf))
        opti.subject_to(opti.bounded(-ego["max_vel"], X[3, k], ego["max_vel"]))

    # 输入约束
    for k in range(N - 1):
        opti.subject_to(opti.bounded(-ego["max_acc"], U[0, k], ego["max_acc"]))
        opti.subject_to(opti.bounded(-ego["max_steer"], U[1, k], ego["max_steer"]))
    for k in range(1, N - 1):
        opti.subject_to(
            opti.bounded(-ego["max_steer_rate"], (U[1, k] - U[1, k - 1]) / (Ts[k] * dt), ego["max_steer_rate"]))
    opti.subject_to(opti.bounded(-ego["max_steer_rate"], (U[1, 0] - u0[1]) / (Ts[0] * dt), ego["max_steer_rate"]))

    # 时间约束
    for k in range(N - 1):
        opti.subject_to(Ts[k] == Ts[k + 1])

    # 车辆边界描述
    g = ca.MX(np.array([[ego["length"] / 2], [ego["width"] / 2], [ego["length"] / 2], [ego["width"] / 2]]))
    G = ca.MX(np.array([[1, 0], [0, -1], [-1, 0], [0, 1]]))
    # 障碍物约束
    for k in range(N):
        Rot = ca.vertcat(ca.horzcat(ca.cos(X[2, k]), -ca.sin(X[2, k])),
                         ca.horzcat(ca.sin(X[2, k]), ca.cos(X[2, k])))
        T = ca.vertcat(X[0, k] + ca.cos(X[2, k]) * ego["rear2center"],
                       X[1, k] + ca.sin(X[2, k]) * ego["rear2center"])

        for m in range(obs_info['onum']):
            oidx_list, vidx_list, A_all, b_all = obs_info['oidx_list'][m], obs_info['vidx_list'][m], obs_info['A'], \
                obs_info['b']
            mu = M[oidx_list[0]:oidx_list[1], k]
            l = L[vidx_list[0]:vidx_list[1], k]
            A = ca.MX(A_all[vidx_list[0]:vidx_list[1], :])
            b = ca.MX(b_all[vidx_list[0]:vidx_list[1]])
            opti.subject_to(-g.T @ mu + (A @ T - b).T @ l > dmin)  # -g'*mu + (A*t - b)*lambda > dmin
            opti.subject_to(G.T @ mu + Rot.T @ A.T @ l == 0)  # G'*mu + R'*A*lambda = 0
            opti.subject_to(ca.norm_2(A.T @ l) <= 1)  # norm(A'*lambda) <= 1
            opti.subject_to(l >= 0)
            opti.subject_to(mu >= 0)

    # 设置初始值
    opti.set_initial(X, ref_traj)
    opti.set_initial(U, ref_input)
    ref_L, ref_M = GetInitialDualVariable(ref_traj, obstacles, ego)
    opti.set_initial(L, ref_L)
    opti.set_initial(M, ref_M)
    opti.set_initial(Ts, dt * np.ones((1, N)))
    # 设置求解器
    options = {'ipopt.max_iter': 2000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
               'ipopt.acceptable_obj_change_tol': 1e-6}
    opti.solver('ipopt')
    sol = opti.solve()
    return sol.value(X), sol.value(U), sol.value(Ts)


if __name__ == '__main__':
    sys.path.append(str(pathlib.Path(__file__).parent.parent))
    from PathPlanning.HybridAStar.hybrid_a_star import *
    from PathPlanning.CubicSpline.spline_continuity import *


    def plot(start, goal, map, obs_list, path=None, trajectory=None):
        global show_animation
        if show_animation:
            if path is None or trajectory is None:
                rs.plot_arrow(start[0, 0], start[1, 0], start[2, 0], fc='g')
                rs.plot_arrow(goal[0, 0], goal[1, 0], goal[2, 0])
                plt.plot(map[0], map[1], "k")
                for obs in obs_list:
                    plt.plot(obs[0], obs[1], 'k')
                plt.grid(True)
                plt.axis("equal")
            else:
                for t_x, t_y, t_yaw in zip(trajectory[0], trajectory[1], trajectory[2]):
                    plt.cla()
                    rs.plot_arrow(start[0, 0], start[1, 0], start[2, 0], fc='g')
                    rs.plot_arrow(goal[0, 0], goal[1, 0], goal[2, 0])
                    plt.plot(map[0], map[1], "k")
                    for obs in obs_list:
                        plt.plot(obs[0], obs[1], 'k')
                    plt.plot(path.x_list, path.y_list, "-k", label="Hybrid A* path")
                    plt.plot(trajectory[0], trajectory[1], "-r", label="OBCA path")
                    plot_car(t_x, t_y, t_yaw)
                    plt.grid(True)
                    plt.axis("equal")
                    plt.pause(0.05)


    def TransferDataForHybridAStar(x0, xF, map, obs_list):
        global XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION
        XY_GRID_RESOLUTION = 0.5
        YAW_GRID_RESOLUTION = np.deg2rad(10.0)

        # 起点，终点格式转换
        start = (x0.T)[0].tolist()[:3]
        goal = (xF.T)[0].tolist()[:3]

        ox, oy = [], []
        # 生成地图边界
        map_x, map_y = [], []
        for i in range(1, len(map[0])):
            x = map[0][i - 1:i + 1]
            y = map[1][i - 1:i + 1]
            sp = Spline2D(x, y, "linear")
            for si in np.arange(0, sp.s[-1], 1.0):
                pi = sp.calc_position(si)
                map_x.append(float(pi[0]))
                map_y.append(float(pi[1]))
        ox.extend(map_x)
        oy.extend(map_y)

        # 生成矩形障碍物
        for obs in obs_list:
            min_x, max_x = min(obs[0]), max(obs[0])
            min_y, max_y = min(obs[1]), max(obs[1])
            x_range = np.arange(min_x, max_x + XY_GRID_RESOLUTION, XY_GRID_RESOLUTION)
            y_range = np.arange(min_y, max_y + XY_GRID_RESOLUTION, XY_GRID_RESOLUTION)
            obs_x, obs_y = [], []
            for x in x_range:
                for y in y_range:
                    obs_x.append(x)
                    obs_y.append(y)
            ox.extend(obs_x)
            oy.extend(obs_y)

        return start, goal, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION


    def GetReferenceTrajectoryFromHybirdAStar(path, ego, x0, dt):
        # # 降采样
        # path.x_list = path.x_list[::2]
        # path.y_list = path.y_list[::2]
        # path.yaw_list = path.yaw_list[::2]

        # # 去除重复节点
        # for i in range(len(path.x_list) - 1, 0, -1):
        #     if (path.x_list[i] == path.x_list[i - 1] and path.y_list[i] == path.y_list[i - 1]):
        #         path.x_list.pop(i)
        #         path.y_list.pop(i)
        #         path.yaw_list.pop(i)

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
        vel_list = []

        if (math.fabs(AngleDiff(math.atan2((y_list[1] - y_list[0]), (x_list[1] - x_list[0])), x0[3, 0])) < np.pi / 2):
            orientation = 1
        else:
            orientation = -1
        vel_list.append(x0[3, 0])

        for i in range(1, len(x_list) - 1):
            p1 = [x_list[i - 1], y_list[i - 1]]
            p2 = [x_list[i], y_list[i]]
            p3 = [x_list[i + 1], y_list[i + 1]]

            if (IsShiftPoint(p2, p1, p3)):
                orientation = -1 * orientation
                vel_list.append(0)
            else:
                vel_list.append(orientation * CalcDistance(p1, p2) / dt)
        vel_list.append(0)

        ref_traj = np.zeros((4, len(x_list)))
        for i in range(len(x_list)):
            ref_traj[0, i] = x_list[i]
            ref_traj[1, i] = y_list[i]
            ref_traj[2, i] = yaw_list[i]
            ref_traj[3, i] = vel_list[i]

        return ref_traj


    def map2obstacle(map):
        obs_list = []
        for i in range(1, len(map)):
            obs_list.append([map[i - 1], map[i]])
        return obs_list


    def list2coord(obj):
        tf_obj = []
        for xy in zip(obj[0], obj[1]):
            tf_obj.append(xy)
        return tf_obj


    dt = 0.2
    u0 = np.array([[0], [0]])
    x0 = np.array([[10], [10.5], [-np.pi], [0]])
    xF = np.array([[0], [1.5], [np.pi / 2], [0]])
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
    map = [[-15, -15, 15, 15, -15],
           [0, 15, 15, 0, 0]]
    obs1 = [[-15, -15, -1.5, -1.5, -15],
            [0, 5, 5, 0, 0]]
    obs2 = [[1.5, 1.5, 15, 15, 1.5],
            [0, 5, 5, 0, 0]]

    plot(x0, xF, map, [obs1, obs2])
    start, goal, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION = TransferDataForHybridAStar(x0, xF, map, [obs1, obs2])
    path = hybrid_a_star_planning(start, goal, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)
    ref_traj = GetReferenceTrajectoryFromHybirdAStar(path, ego, x0, dt)
    ref_input = CalculateReferenceInput(ref_traj, ego)

    opt_traj, opt_input, _ = planning(x0, xF, u0, ego, list2coord(map), [list2coord(obs1), list2coord(obs2)], ref_traj,
                                      ref_input, dt)
    plot(x0, xF, map, [obs1, obs2], path, opt_traj)
