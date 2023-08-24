import numpy as np
import dill
import time
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import csv

import linkleg_transform as lt
import LegKinematics as lk
import simple_model_ode as sm
from Bezier import Bezier


MODE_STOP = 0
MODE_STANCE = 1
MODE_WALK = 2
MODE_TROT = 3

color_modlist = ["cornflowerblue", "forestgreen", "darkorange", "plum"]
color_body = ["darkred", "indianred"]


class FiniteStateMachine:
    def __init__(self, loop_freq):
        # list order -> Mod_A, Mod_B, Mod_C, Mod_D
        # True -> Swing, False -> Stance
        self.contact = [False, False, False, False]
        self.swing_counts = [0, 0, 0, 0]
        self.trigger_round = [1, 3, 2, 4]
        # self.trigger_round = [4, 1, 3, 2]
        self.count = 0
        self.mode = MODE_STOP

        self.timer_trigger_time = 0
        self.timer_count = 0
        self.frequency = loop_freq

    def update(self, mode, swing_time):
        self.mode = mode
        swing_count = swing_time * self.frequency
        index = self.trigger_round[self.timer_trigger_time % 4] - 1

        self.timer_count -= 1

        if self.timer_count < 0:
            if mode == MODE_STOP:
                pass
            elif mode == MODE_STANCE:
                pass
            elif mode == MODE_TROT:
                self.swing_counts[index] = swing_count
                self.timer_count = 0 if index < 2 else swing_count
                self.timer_trigger_time += 1
                pass
            elif mode == MODE_WALK:
                self.swing_counts[index] = swing_count
                self.timer_count = swing_count
                self.timer_trigger_time += 1
                pass

        for i in range(4):
            self.swing_counts[i] = self.swing_counts[i] - 1

            if self.swing_counts[i] >= 0:
                self.contact[i] = True
            else:
                self.contact[i] = False

        self.count += 1


class LinkLeg:
    def __init__(self, idx, freq, tb):
        self.idx = idx
        self.freq = freq
        self.state_tb = tb
        self.state_phi = lt.getPhiRL(tb)
        self.vec_OG = lk.FowardKinematics(tb)
        # self.vec_OG = vec_OG(self.state_phi[0, 0], self.state_phi[1, 0])
        self.len_OG = np.linalg.norm(self.vec_OG)
        self.LDM = sm.SimplifiedModel(data_analysis=False)

        # quadratic Trajectory coeff
        self.swingTrajectory = [0, 0, 0]

        # For inverse dynamics iteration
        self.prev_state_tb = [
            self.state_tb[0, 0],
            0,
            0,
            self.state_tb[1, 0],
            0,
            0,
        ]  # [theta, dtheta, ddtheta, beta, dbeta, ddbeta]
        self.cur_state_tb = self.prev_state_tb.copy()

    def updateState(self, tb):
        self.state_tb = tb
        self.state_phi = lt.getPhiRL(tb)
        # self.vec_OG = vec_OG(self.state_phi[0], self.state_phi[1])
        # self.len_OG = np.linalg.norm(self.vec_OG)

    def moveStance(self, vel):
        if self.idx == 0 or self.idx == 3:
            vel *= -1
            dbeta_sign = -1
        else:
            dbeta_sign = 1
        dx = vel / self.freq
        vec_dx = np.array([[dx], [0]])
        vec_OG_k = self.vec_OG - vec_dx
        leglen_k = np.linalg.norm(vec_OG_k)

        # get theta_k from inverse kinematics approx...
        theta_k = lk.InverseKinematicsPoly(np.array([[0], [-leglen_k]]))[0, 0]
        dbeta_k = dbeta_sign * np.arccos((vec_OG_k.T @ self.vec_OG) / (self.len_OG * leglen_k))[0, 0]

        # Update
        self.len_OG = leglen_k
        self.vec_OG = vec_OG_k
        beta_k = self.state_tb[1, 0] + dbeta_k
        self.state_tb = np.array([[theta_k], [beta_k]])
        self.state_phi = lt.getPhiRL(self.state_tb)

    def moveSwing(self, sp):
        # move leg to next swingpoint
        l_des = np.linalg.norm(sp)
        tb_ = lk.InverseKinematicsPoly(sp)
        theta = tb_[0, 0]
        beta = tb_[1, 0]
        self.updateState(np.array([[theta], [beta]]))
        self.vec_OG = sp
        self.len_OG = l_des

    def updateDynamicState(self, dt):
        self.cur_state_tb[0] = self.state_tb[0, 0]  # theta
        self.cur_state_tb[1] = (self.cur_state_tb[0] - self.prev_state_tb[0]) / dt  # dtheta
        self.cur_state_tb[2] = (self.cur_state_tb[1] - self.prev_state_tb[1]) / dt  # ddtheta
        self.cur_state_tb[3] = self.state_tb[1, 0]  # beta
        self.cur_state_tb[4] = (self.cur_state_tb[3] - self.prev_state_tb[3]) / dt  # dbeta
        self.cur_state_tb[5] = (self.cur_state_tb[4] - self.prev_state_tb[4]) / dt
        self.prev_state_tb = self.cur_state_tb.copy()

    def resetDynamicState(self):
        self.prev_state_tb[0] = self.state_tb[0, 0]  # theta
        self.prev_state_tb[1] = 0
        self.prev_state_tb[2] = 0
        self.prev_state_tb[3] = self.state_tb[1, 0]  # beta
        self.prev_state_tb[4] = 0
        self.prev_state_tb[5] = 0
        self.cur_state_tb = self.prev_state_tb.copy()

    def getSwingLegInvDynamics(self):
        # LDM.inverseLinkLegODE
        F_rm, T_b = self.LDM.inverseLinkLegODE(self.cur_state_tb)
        F_x = -1 * abs(F_rm) * np.sin(self.cur_state_tb[0])
        F_y = -1 * abs(F_rm) * np.cos(self.cur_state_tb[3])

        if self.idx == 0 or self.idx == 3:
            F_x *= -1
            T_bf = np.array([[0], [-T_b], [0]])
        else:
            T_bf = np.array([[0], [T_b], [0]])

        F_bf = np.array([[F_x], [0], [F_y]])
        return [F_bf, T_bf]


class Corgi:
    def __init__(self, loop_freq, cpg):
        self.cpg = cpg
        self.mode = MODE_WALK
        self.step_length = 0.2
        self.step_height = 0.05
        # swing_time = step_length/heading_velocity
        self.swing_time = 0.4
        self.stance_height = 0.2
        self.average_vel = 0
        self.total_time = 10
        self.total_cycle = 4 * 4

        # Robot initial state (in Theta-Beta Representation)
        self.initial_position = np.array([[0], [0], [0.1]])
        self.Position = self.initial_position
        self.Orientation = np.array([0, 0, 0, 1])  # xyzw

        # Physics Prop. (SI unit)
        self.g = np.array([[0], [0], [-9.80665]])
        self.m = 26.662009  # kg
        self.m_leg = 0.654
        self.m_body = self.m - self.m_leg
        self.d_l = 577.5 * 0.001
        self.d_w = 329.5 * 0.001
        self.d_h = 144 * 0.001
        self.d_shaft = 0.444
        self.I_zz = 1 / 12 * self.m * (self.d_l**2 + self.d_w**2)
        self.I_xx = 1 / 12 * self.m * (self.d_h**2 + self.d_w**2)
        self.I_yy = 1 / 12 * self.m * (self.d_h**2 + self.d_l**2)
        self.I = np.array([[self.I_xx, 0, 0], [0, self.I_yy, 0], [0, 0, self.I_zz]])

        # Module initial state (in Theta-Beta Representation)
        self.leg_A = LinkLeg(0, loop_freq, np.array([[np.deg2rad(17)], [0]]))
        self.leg_B = LinkLeg(1, loop_freq, np.array([[np.deg2rad(17)], [0]]))
        self.leg_C = LinkLeg(2, loop_freq, np.array([[np.deg2rad(17)], [0]]))
        self.leg_D = LinkLeg(3, loop_freq, np.array([[np.deg2rad(17)], [0]]))
        self.vec_C_AO = np.array([[self.d_shaft / 2], [self.d_w / 2], [0]])
        self.vec_C_BO = np.array([[self.d_shaft / 2], [-self.d_w / 2], [0]])
        self.vec_C_CO = np.array([[-self.d_shaft / 2], [-self.d_w / 2], [0]])
        self.vec_C_DO = np.array([[-self.d_shaft / 2], [self.d_w / 2], [0]])

        self.legs = [self.leg_A, self.leg_B, self.leg_C, self.leg_D]
        self.legs_offset = [self.vec_C_AO, self.vec_C_BO, self.vec_C_CO, self.vec_C_DO]

        self.frequency = loop_freq
        self.dt = 1 / loop_freq

        self.iter_t = 0
        self.loop_cnt = 0
        self.cycle_cnt = 0
        self.Trajectory = []
        self.performances = []
        self.t_list = []

        # Optimization
        self.cost = 0
        self.weight_s = 1
        self.weight_st = 0.1
        self.weight_u = 1
        self.weight_R1 = 5
        self.weight_R2 = 5
        self.weight_L1 = 5
        self.pot_wall_thres = 0.01

        # Animation
        self.animate_fps = 60
        self.ax1 = None
        self.ax2 = None
        self.vs_com = None
        self.vs_com_proj = None
        self.vs_fhs = []
        self.vs_stab = None
        self.vs_sup_polygon = None
        self.vs_stab_line_t = []
        self.vs_stab_line_s = []

    def standUp(self, vel):
        # print("Standing Up ...")
        tolerance = 0.01
        t = time.time()
        while np.abs(self.stance_height - self.Position[2, 0]) > tolerance:
            for leg in self.legs:
                l_des = leg.len_OG + vel * self.dt
                theta_ = lk.InverseKinematicsPoly(np.array([[0], [-l_des]]))[0, 0]
                leg.updateState(np.array([[theta_], [0]]))
                leg.len_OG += vel * self.dt
                leg.vec_OG[1, 0] -= vel * self.dt

            self.Position = self.Position + np.array([[0], [0], [vel * self.dt]])
            self.recordTrajectory()
            self.t_list.append(self.iter_t)
            self.performances.append([-1, 0])
            self.iter_t += self.dt

        # print("Time Elapsed = ", time.time() - t)
        # print("iter_t = ", self.iter_t)

    def move(self, swing_mode="Bezier", swing_profile="default"):
        prev_contact = [False, False, False, False]
        t_sw = 0
        quadratic_config = []
        bezier_config = []
        liftoff_leg = -1
        F_fl = np.array([[0], [0], [0]])  # Flight Leg Force
        T_ffl = np.array([[0], [0], [0]])
        T_fl = np.array([[0], [0], [0]])  # Flight Leg Torque
        s = 0  # FA stability
        swing_point = 0

        while self.loop_cnt < self.total_time * self.frequency:
            self.cpg.update(self.mode, self.swing_time)
            self.Position += np.array([[self.average_vel * self.dt], [0], [0]])

            for i in range(4):
                if self.cpg.contact[i] == False:
                    # Stance Phase
                    self.legs[i].moveStance(self.average_vel)
                else:
                    # Flight Phase
                    if self.cpg.contact != prev_contact:
                        # Plan lift off trajectory
                        t_sw = self.dt
                        # print("Lift ", i, "\n---")
                        # print("lp_hip", self.getHipPosition(i, self.Position).T)
                        liftoff_leg = i
                        # quadratic_config = self.getQuadraticTrajectory(i)
                        # self.average_vel = self.getAverageVelocity(quadratic_config[1], i)[0, 0]

                        if isinstance(swing_profile, str):
                            bezier_config = self.getBezierTrajectory(
                                i, bezier_profile=[0.08, 0.01, 0.08, -0.04, 0.08, -0.04]
                            )
                            # bezier_config = self.getBezierTrajectory(
                            #     i, bezier_profile=[0.06, 0.01, 0.1, -0.04, 0.1, -0.04]
                            # )

                        else:
                            bez_profile = swing_profile[i]
                            bezier_config = self.getBezierTrajectory(i, bezier_profile=bez_profile)

                        self.average_vel = self.getAverageVelocity(
                            bezier_config[2] + np.array([[self.step_length], [0], [0]]), i
                        )[0, 0]
                        self.legs[i].resetDynamicState()
                        prev_contact = self.cpg.contact.copy()
                        self.cycle_cnt += 1

                    self.legs[i].updateDynamicState(self.dt)

                    if swing_mode == "Bezier":
                        swing_point = self.getBezierSwingPoint(t_sw, bezier_config)
                    elif swing_mode == "Quadratic":
                        swing_point = self.getQuadraticSwingPoint(
                            t_sw, quadratic_config[0], quadratic_config[2], quadratic_config[3]
                        )

                    bf_sp = swing_point - self.Position - self.legs_offset[i]
                    lf_sp = self.transformBF2LF(i, bf_sp)
                    try:
                        self.legs[i].moveSwing(lf_sp)
                    except:
                        print("p_hip", self.getHipPosition(i, self.Position).T)
                        print("swing_point", swing_point.T)
                        self.legs[i].moveSwing(lf_sp)

                    F_fl, T_fl = self.legs[i].getSwingLegInvDynamics()
                    # Torque Cause by Swinging Force
                    T_ffl = np.cross(self.legs_offset[i].T, F_fl.T).T
                    tau_RL = lt.getTauRL(np.array([[F_fl], [T_fl]]), self.legs[i].state_tb[0, 0])
                    # CR1 = self.getPotentialCostR1(0.1, swing_point, self.getHipPosition(i, self.Position))
                    # CR2 = self.getPotentialCostR2(0.34290456, swing_point, self.getHipPosition(i, self.Position))
                    # CL1 = self.getPotentialCostL1(i, self.Position, swing_point)

            if self.cycle_cnt > self.total_cycle:
                self.cost += self.weight_st * s
                print("cost", self.cost)
                break

            Fr_ = np.array([[0], [0], [-9.81 * self.m_body]]) + F_fl
            Nr_ = np.array([[0], [0], [0]]) + T_fl + T_ffl
            s = self.evaluateFAStability(Fr_, Nr_)

            # Integrate Cost for optimization
            self.cost += (self.weight_s * -s + self.weight_u * (tau_RL.T @ tau_RL)[0, 0]) * self.dt
            # self.cost += (CR1 + CR2 + CL1) * self.dt

            self.performances.append([liftoff_leg, s])
            # self.performances.append([liftoff_leg, self.cost])
            self.t_list.append(self.iter_t)

            self.recordTrajectory()
            self.iter_t += self.dt
            t_sw += self.dt
            self.loop_cnt += 1

        # Integrate terminal cost
        # self.cost += self.weight_st * s

    def transformBF2LF(self, idx, p_bf):
        if idx == 0 or idx == 3:
            p_lf = np.array([[-1, 0, 0], [0, 0, 1]]) @ p_bf
        else:
            p_lf = np.array([[1, 0, 0], [0, 0, 1]]) @ p_bf
        return p_lf

    def transformLF2BF(self, idx, p_lf):
        if idx == 0 or idx == 3:
            p_bf = np.array([[-1, 0], [0, 0], [0, 1]]) @ p_lf
        else:
            p_bf = np.array([[1, 0], [0, 0], [0, 1]]) @ p_lf
        return p_bf

    def getHipPosition(self, idx, com_pose):
        return com_pose + self.legs_offset[idx]

    def getQuadraticTrajectory(self, leg_idx):
        leg = self.legs[leg_idx]
        v_OG = self.transformLF2BF(leg_idx, leg.vec_OG)

        leg_cp1 = self.Position + self.legs_offset[leg_idx] + v_OG
        leg_cp2 = leg_cp1 + np.array([[self.step_length], [0], [0]])

        # quadratic trajectory y(t) = cff*t(t-T)
        cff_x = self.step_length / self.swing_time
        cff_y = -2 * self.step_height / (self.swing_time**2)
        return [leg_cp1, leg_cp2, cff_x, cff_y]

    def getQuadraticSwingPoint(self, t, cp1, cff_x, cff_y):
        # Return Foothold point of swing foot in world frame
        dx = cff_x * t
        dz = cff_y * t * (t - self.swing_time)
        return cp1 + np.array([[dx], [0], [dz]])

    def getBezierTrajectory(self, leg_idx, bezier_profile, point_num=100):
        leg = self.legs[leg_idx]
        v_OG = self.transformLF2BF(leg_idx, leg.vec_OG)

        leg_cp1 = self.Position + self.legs_offset[leg_idx] + v_OG
        leg_cp2 = leg_cp1 + np.array([[self.step_length], [0], [0]])
        cp1 = np.array([leg_cp1[0, 0], leg_cp1[2, 0]])
        cp2 = np.array([leg_cp2[0, 0], leg_cp2[2, 0]])

        # 12 Control Points of bezier curves [REF]
        h, dh, dL1, dL2, dL3, dL4 = bezier_profile
        L = self.step_length

        c0 = cp1
        c1 = c0 - np.array([dL1, 0])
        c2 = c1 - np.array([dL2, 0]) + np.array([0, h])
        c3 = c2
        c4 = c2
        c5 = c4 + np.array([0.5 * L + dL1 + dL2, 0])
        c6 = c5
        c7 = c5 + np.array([0, dh])
        c8 = c7 + np.array([0.5 * L + dL3 + dL4, 0])
        c9 = c8
        c10 = c8 - np.array([dL4, h + dh])
        c11 = c10 - np.array([dL3, 0])
        c_set = np.array([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11])

        t_points = np.linspace(0, 1, point_num)
        curve = Bezier.Curve(t_points, c_set)

        return [t_points, curve, leg_cp1]

    def getBezierSwingPoint(self, t, config):
        ts, curve, cp1 = config
        t_ = t / self.swing_time
        x = np.interp(t_, ts, curve[:, 0])
        z = np.interp(t_, ts, curve[:, 1])
        # print("swp", (cp1 + np.array([[x], [0], [z]])).T)
        return np.array([[x], [cp1[1, 0]], [z]])

    def bezierValidate(self, idx, p_hip, curve):
        valid = True
        p_c = p_hip - self.legs_offset[idx]
        for p in curve:
            p_ = np.array([[p[0]], [p_hip[1, 0]], [p[1]]])
            """ print("p_hip", p_hip)
            print("p_", p_)
            print("p_c", p_c)
            print("offset", self.legs_offset[idx]) """
            d = np.linalg.norm(p_ - p_hip)
            if d < 0.10031048 or d > 0.34290456:
                valid = False
                print("length validation")
            if idx == 0 or idx == 1:
                if p_[0, 0] < p_c[0, 0]:
                    valid = False
                    print("mid validation")
                    break
            else:
                if p_[0, 0] > p_c[0, 0]:
                    valid = False
                    print("mid validation")
                    break
        return valid

    def getPotentialCostR1(self, R_bound, wf_sp, p_hip):
        r_ = np.linalg.norm(wf_sp - p_hip)
        Q_ = R_bound + self.pot_wall_thres  # potential effective outer ring
        if r_ <= Q_:
            U = 0.5 * self.weight_R1 * (1 / r_ - 1 / Q_) ** 2
        else:
            U = 0
        return U

    def getPotentialCostR2(self, R_bound, wf_sp, p_hip):
        r_ = np.linalg.norm(wf_sp - p_hip)
        # print("wf_sp", wf_sp)
        # print("p_hip", p_hip)
        D_ = abs(r_ - R_bound)
        D_min_ = 0.0001
        if D_ < D_min_:
            D_ = D_min_
        Q_ = self.pot_wall_thres
        if r_ < R_bound:
            if D_ <= self.pot_wall_thres:
                U = 0.5 * self.weight_R2 * (1 / D_ - 1 / Q_) ** 2
            else:
                U = 0
        else:
            U = 0.5 * self.weight_R2 * (1 / D_min_ - 1 / Q_) ** 2

        return U

    def getPotentialCostL1(self, idx, p_com, wf_sp):
        r_ = wf_sp[0, 0] - p_com[0, 0]
        r_min_ = 0.0001
        Q_ = self.pot_wall_thres

        if abs(r_) < r_min_:
            r_ = r_min_ * np.sign(r_)

        if idx == 0 or idx == 1:
            if abs(r_) >= 0:
                if r_ <= Q_:
                    U = 0.5 * self.weight_L1 * (1 / r_ - 1 / Q_) ** 2
                else:
                    U = 0
            else:
                U = 0.5 * self.weight_L1 * (1 / r_min_ - 1 / Q_) ** 2
        else:
            if r_ <= 0:
                if abs(r_) <= Q_:
                    U = 0.5 * self.weight_L1 * (1 / abs(r_) - 1 / Q_) ** 2
                else:
                    U = 0
            else:
                U = 0.5 * self.weight_L1 * (1 / r_min_ - 1 / Q_) ** 2
        return U

    def getAverageVelocity(self, cp2, lift_idx):
        cps1 = np.array([[0], [0], [0]])
        cps2 = np.array([[0], [0], [0]])

        for i in range(4):
            v_OG = self.transformLF2BF(i, self.legs[i].vec_OG)

            cp = self.Position + self.legs_offset[i] + v_OG
            if i != lift_idx:
                cps1 = cps1 + cp
                cps2 = cps2 + cp
            else:
                cps1 = cps1 + cp
                cps2 = cps2 + cp2

        com1 = 1 / 4 * cps1
        com2 = 1 / 4 * cps2
        vel = (com2 - com1) / self.swing_time
        return vel

    def evaluateFAStability(self, Fr, Nr):
        contact_legs = []
        contact_point_bf = []
        stability_idxs = []
        # p_c_ = self.getCOMPosition()
        p_c_ = self.Position
        if self.cpg.mode == MODE_WALK:
            for i in range(4):
                if self.cpg.contact[i] == False:
                    contact_legs.append(self.legs[i])
                    contact_point_bf.append(self.getFootTipPosition(i))
            v_a0 = contact_point_bf[1] - contact_point_bf[0]
            v_a1 = contact_point_bf[2] - contact_point_bf[1]
            v_a2 = contact_point_bf[0] - contact_point_bf[2]
            # side_vec = [v_a0, v_a1, v_a2]
            u_a0 = v_a0 / np.linalg.norm(v_a0)
            u_a1 = v_a1 / np.linalg.norm(v_a1)
            u_a2 = v_a2 / np.linalg.norm(v_a2)
            uside_vec = [u_a0, u_a1, u_a2]  # unit vector of support triangle

            for i in range(3):
                # Evaluate Force Angle Stability for each side of support triangle
                u_s_ = uside_vec[i]
                # s_ = side_vec[i]
                p_i_ = contact_point_bf[i]
                l_ = (np.eye(3) - u_s_ @ u_s_.T) @ (p_i_ - p_c_)
                F_ = (np.eye(3) - u_s_ @ u_s_.T) @ Fr
                N_ = u_s_ @ u_s_.T @ Nr
                # Equiv. Torque to Force Couple
                F_n_ = np.cross(l_.T, N_.T).T / np.linalg.norm(l_)
                F_ = F_ + F_n_

                u_F_ = F_ / np.linalg.norm(F_)
                d_ = -l_ + (l_.T @ u_F_)[0, 0] * u_F_
                if (np.cross(F_.T, l_.T) @ u_s_)[0, 0] >= 0:
                    sign = 1
                else:
                    sign = -1

                if abs((F_.T @ l_) / (np.linalg.norm(F_) * np.linalg.norm(l_)) - 1) < 0.0001:
                    angle_ = np.array([[0]])
                else:
                    angle_ = sign * np.arccos((F_.T @ l_) / (np.linalg.norm(F_) * np.linalg.norm(l_)))
                stab_ = angle_ * np.linalg.norm(d_) * np.linalg.norm(F_)
                stability_idxs.append(stab_[0, 0])
            stability_idx = min(stability_idxs)
            # print("stability idx", stability_idx)
            return stability_idx

        elif self.cpg.mode == MODE_TROT:
            pass

    def getFootTipPosition(self, idx):
        # Return foot tip position in World Frame
        lf_OG = lk.FowardKinematics(self.legs[idx].state_tb)
        if idx == 0 or idx == 3:
            v_OG = np.array([[-1, 0], [0, 0], [0, 1]]) @ lf_OG
        else:
            v_OG = np.array([[1, 0], [0, 0], [0, 1]]) @ lf_OG
        return self.Position + self.legs_offset[idx] + v_OG

    def getCOMPosition(self):
        # return com position considering link leg mass in Base Frame
        coms_ = np.array([[0], [0], [0]])
        for i, leg in enumerate(self.legs):
            rm_ = lt.getRm(leg.state_tb[0, 0])
            lf_OG = leg.vec_OG
            if i == 0 or i == 3:
                v_OG = np.array([[-1, 0], [0, 0], [0, 1]]) @ lf_OG
            else:
                v_OG = np.array([[1, 0], [0, 0], [0, 1]]) @ lf_OG
            coms_ += self.m_leg * (self.Position + self.legs_offset[i] + (rm_ / np.linalg.norm(v_OG)) * v_OG)
        coms_ += self.m_body * self.Position
        coms_ = coms_ / 4 * self.m
        return coms_

    def recordTrajectory(self):
        leg_state = [self.iter_t, self.Position.reshape(1, -1).ravel().tolist()]
        for leg in self.legs:
            leg_state.append(leg.state_tb.reshape(1, 2).ravel().tolist())
            leg_state.append(leg.state_phi.reshape(1, 2).tolist()[0])

        leg_state.append(copy.copy(self.cpg.contact))
        self.Trajectory.append(leg_state)
        """ leg_state = [t, [px, py, pz], [MA_t, MA_b], [MA_phiR, MA_phiL],
                                      [MB_t, MB_b], [MB_phiR, MB_phiL],
                                      [MC_t, MC_b], [MC_phiR, MC_phiL],
                                      [MD_t, MD_b], [MD_phiR, MD_phiL],
                                      [MOD_A_contact, MOD_B_contact, MOD_C_contact, MOD_D_contact]] """
        pass

    def updateAnimation(self, frame_cnt):
        idx = round(frame_cnt * (self.frequency / self.animate_fps))

        if idx <= len(self.Trajectory):
            p_com = self.Trajectory[idx][1]
            p_com_vec = np.array([[p_com[0]], [p_com[1]], [p_com[2]]])
            self.vs_com.set_data(p_com[0], p_com[1])
            self.vs_com.set_3d_properties(p_com[2])

            self.vs_stab_line_t.append(self.Trajectory[idx][0])
            self.vs_stab_line_s.append(self.performances[idx][1])
            self.vs_stab.set_data([self.vs_stab_line_t, self.vs_stab_line_s])
            # self.vs_stab.set_color(color_modlist[self.performances[idx][0]])
            self.vs_stab.set_color("gray")
            self.vs_stab.set_alpha(0.2)
            self.ax2.scatter(
                self.Trajectory[idx][0], self.performances[idx][1], color=color_modlist[self.performances[idx][0]]
            )

            p_A_hip = self.getHipPosition(0, p_com_vec)
            p_B_hip = self.getHipPosition(1, p_com_vec)
            p_C_hip = self.getHipPosition(2, p_com_vec)
            p_D_hip = self.getHipPosition(3, p_com_vec)
            p_hip = [p_A_hip, p_B_hip, p_C_hip, p_D_hip]

            sup_x = []
            sup_y = []
            sup_z = []
            for i in range(4):
                midx = 2 * i + 3
                phiR_, phiL_ = self.Trajectory[idx][midx]
                lf_OG = lk.FowardKinematics(np.array([[phiR_], [phiL_]]), "phi")
                if i == 0 or i == 3:
                    v_OG = np.array([[-1, 0], [0, 0], [0, 1]]) @ lf_OG
                else:
                    v_OG = np.array([[1, 0], [0, 0], [0, 1]]) @ lf_OG

                p_G = p_hip[i] + v_OG
                self.vs_fhs[i].set_data([p_hip[i][0, 0], p_G[0, 0]], [p_hip[i][1, 0], p_G[1, 0]])
                self.vs_fhs[i].set_3d_properties([p_hip[i][2, 0], p_G[2, 0]])

                if abs(p_G[2, 0]) < 0.001:
                    sup_x.append(p_G[0, 0])
                    sup_y.append(p_G[1, 0])
                    sup_z.append(p_G[2, 0])
            sup_x.append(sup_x[0])
            sup_y.append(sup_y[0])
            sup_z.append(sup_z[0])
            self.vs_sup_polygon.set_data(sup_x, sup_y)
            self.vs_sup_polygon.set_3d_properties(sup_z)
            self.vs_com_proj.set_data(p_com[0], p_com[1])
            self.vs_com_proj.set_3d_properties(0)

        return (self.vs_com,)

    def resetVisualize(self):
        self.vs_stab_line_s = []
        self.vs_stab_line_t = []

        self.ax2.clear()
        self.ax2.axes.set_xlim([0, self.Trajectory[-1][0]])
        self.ax2.axes.set_ylim([-6, 6])
        self.vs_stab = self.ax2.plot([], [], "o-")[0]

    def visualize(self):
        # FPS <= loop frequency
        frame_interval = round((1 / self.animate_fps) * 1000)
        frame_count = round(self.Trajectory[-1][0] * self.animate_fps)
        print("Frame interval count:", frame_interval, frame_count)

        fig = plt.figure(figsize=(16, 18), dpi=50)

        # self.ax = Axes3D(fig)
        # fig.add_axes(self.ax)
        self.ax1 = fig.add_subplot(2, 1, 1, projection="3d")
        self.ax1.set_xlabel("X")
        self.ax1.set_ylabel("Y")
        self.ax1.set_zlabel("Z")
        self.ax1.axes.set_xlim3d([-0.2, 1.2])
        self.ax1.axes.set_ylim3d([-0.5, 0.5])
        self.ax1.axes.set_zlim3d([0, 0.8])
        sf_x = np.arange(-1, 5, 0.25)
        sf_y = np.arange(-1, 1, 0.25)
        X, Y = np.meshgrid(sf_x, sf_y)
        self.ax1.plot_surface(X, Y, 0 * X, alpha=0.1, color="gray")

        self.vs_com = self.ax1.plot([], [], [], "o", color=color_body[1])[0]
        for i in range(4):
            self.vs_fhs.append(self.ax1.plot([], [], [], "o-", color=color_modlist[i])[0])
        self.vs_sup_polygon = self.ax1.plot([], [], [], "o-", color=color_body[1])[0]
        self.vs_com_proj = self.ax1.plot([], [], [], "D", color="orange")[0]

        self.ax2 = fig.add_subplot(2, 1, 2)
        self.ax2.axes.set_xlim([0, self.Trajectory[-1][0]])
        self.ax2.axes.set_ylim([-6, 6])
        self.ax2.axes.set_xlabel("time [sec]")
        self.ax2.axes.set_ylabel("Force Angle Stability")
        self.vs_stab = self.ax2.plot([], [], color="gray")[0]

        ani = animation.FuncAnimation(
            fig,
            self.updateAnimation,
            frames=frame_count,
            interval=frame_interval,
            repeat=True,
            init_func=self.resetVisualize(),
        )

        # ani.save("bezier.mp4", fps=self.animate_fps)
        plt.grid()
        plt.show()

    def exportCSV(self, filepath="./csv_trajectory/output.csv", mode="sbrio"):
        with open(filepath, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            for i in self.Trajectory:
                if mode == "sbrio":
                    A_phiRL = lt.getPhiRL(np.array([[i[2][0]], [-i[2][1]]]))
                    D_phiRL = lt.getPhiRL(np.array([[i[8][0]], [-i[8][1]]]))
                else:
                    A_phiRL = lt.getPhiRL(np.array([[i[2][0]], [i[2][1]]]))
                    D_phiRL = lt.getPhiRL(np.array([[i[8][0]], [i[8][1]]]))
                A_phiR = A_phiRL[0, 0]
                A_phiL = A_phiRL[1, 0]
                B_phiR = i[5][0]
                B_phiL = i[5][1]
                C_phiR = i[7][0]
                C_phiL = i[7][1]
                D_phiR = D_phiRL[0, 0]
                D_phiL = D_phiRL[1, 0]
                A_contact = i[10][0]
                B_contact = i[10][1]
                C_contact = i[10][2]
                D_contact = i[10][3]
                row = [
                    A_phiR,
                    A_phiL,
                    B_phiR,
                    B_phiL,
                    C_phiR,
                    C_phiL,
                    D_phiR,
                    D_phiL,
                    A_contact,
                    B_contact,
                    C_contact,
                    D_contact,
                ]
                writer.writerow(row)


if __name__ == "__main__":
    print("CPG Started")

    loop_freq = 1000  # Hz
    FSM = FiniteStateMachine(loop_freq)
    CORGI = Corgi(loop_freq, FSM)
    LDM = sm.SimplifiedModel(data_analysis=False)

    CORGI.standUp(0.05)
    CORGI.move()
    CORGI.visualize()
    CORGI.exportCSV()
