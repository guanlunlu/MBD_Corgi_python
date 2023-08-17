import numpy as np
import dill
import time
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

import linkleg_transform as lt
import LegKinematics as lk
import simple_model_ode as sm


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
        # self.trigger_round = [1, 3, 2, 4]
        self.trigger_round = [4, 1, 3, 2]
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

        # quadratic Trajectory coeff
        self.swingTrajectory = [0, 0, 0]

        # For inverse dynamics iteration
        self.prev_state = [
            self.state_tb[0, 0],
            0,
            0,
            self.state_tb[1, 0],
            0,
            0,
        ]  # [theta, dtheta, ddtheta, beta, dbeta, ddbeta]
        self.cur_state = self.prev_state

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
        theta_k = self.solveLegIK(leglen_k, self.state_tb)

        dbeta_k = (
            dbeta_sign
            * np.arccos((vec_OG_k.T @ self.vec_OG) / (self.len_OG * leglen_k))[0, 0]
        )

        # Update
        self.len_OG = leglen_k
        self.vec_OG = vec_OG_k
        beta_k = self.state_tb[1, 0] + dbeta_k
        self.state_tb = np.array([[theta_k], [beta_k]])
        self.state_phi = lt.getPhiRL(self.state_tb)

    def moveSwing(self, sp):
        # move leg to next swingpoint
        l_des = np.linalg.norm(sp)
        theta = self.solveLegIK(l_des, self.state_tb)

        u_sp = sp / l_des
        beta = np.arccos(-u_sp.T @ np.array([[0], [1]]))[0, 0]
        if sp[0, 0] <= 0:
            beta_sign = 1
        else:
            beta_sign = -1
        beta = abs(beta) * beta_sign

        self.updateState(np.array([[theta], [beta]]))
        self.vec_OG = sp
        self.len_OG = l_des

    def solveLegIK(self, desired_length, current_tb):
        # Solve theta correspond to desired length
        if desired_length > 0.3428:
            return "error"
        else:
            # phiRL = lt.getPhiRL(current_tb)
            # v_OG = vec_OG(phiRL[0, 0], phiRL[1, 0])
            # iter_length = np.sqrt(v_OG.T @ v_OG)[0, 0]
            v_OG = lk.FowardKinematics(current_tb)
            iter_length = np.linalg.norm(v_OG)
            theta = current_tb[0, 0]

            error = desired_length - iter_length

            tolerance = 0.0001
            p_gain = 10

            while abs(error) > tolerance:
                theta = theta + (p_gain * error)
                # phiRL = lt.getPhiRL(np.array([[theta], [0]]))
                # v_OG = vec_OG(phiRL[0, 0], phiRL[1, 0])
                v_OG = lk.FowardKinematics(np.array([[theta], [0]]))
                iter_length = np.sqrt(v_OG.T @ v_OG)[0, 0]
                error = desired_length - iter_length
            return theta

    def updateDynamicState(self, dt):
        self.cur_state[0] = self.state_tb[0, 0]  # theta
        self.cur_state[1] = (self.cur_state[0] - self.prev_state[0]) / dt  # dtheta
        self.cur_state[2] = (self.cur_state[1] - self.prev_state[1]) / dt  # ddtheta
        self.cur_state[3] = self.state_tb[1, 0]  # beta
        self.cur_state[4] = (self.cur_state[3] - self.prev_state[3]) / dt  # dbeta
        self.cur_state[5] = (self.cur_state[4] - self.prev_state[4]) / dt

    def getSwingLegInvDynamics(self):
        # LDM.inverseLinkLegODE
        F_rm, T_b = LDM.inverseLinkLegODE(self.cur_state)
        F_x = -1 * abs(F_rm) * np.sin(self.cur_state[0])
        F_y = -1 * abs(F_rm) * np.cos(self.cur_state[3])

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
        self.step_height = 0.1
        # swing_time = step_length/heading_velocity
        self.swing_time = 0.8
        self.stance_height = 0.2
        self.total_time = 10

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
        self.swing_cff_x = 0  # Coefficient of Swing phase quadratic trajectory
        self.swing_cff_y = 0  # Coefficient of Swing phase quadratic trajectory
        self.Trajectory = []
        self.performances = []
        self.t_list = []

        # Animation
        self.animate_fps = 30
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
        print("Standing Up ...")
        tolerance = 0.01
        t = time.time()
        while np.abs(self.stance_height - self.Position[2, 0]) > tolerance:
            for leg in self.legs:
                theta_ = leg.solveLegIK(leg.len_OG + vel * self.dt, leg.state_tb)
                leg.updateState(np.array([[theta_], [0]]))
                leg.len_OG += vel * self.dt
                leg.vec_OG[1, 0] -= vel * self.dt

            self.Position = self.Position + np.array([[0], [0], [vel * self.dt]])
            self.recordTrajectory()
            self.t_list.append(self.iter_t)
            self.performances.append(0)
            self.iter_t += self.dt

        print("Time Elapsed = ", time.time() - t)
        print("iter_t = ", self.iter_t)

    def move(self):
        prev_contact = [False, False, False, False]
        avg_vel = 0
        t_sw = 0
        swing_config = []
        F_fl = np.array([[0], [0], [0]])  # Flight Leg Force
        T_ffl = np.array([[0], [0], [0]])
        T_fl = np.array([[0], [0], [0]])  # Flight Leg Torque

        while self.loop_cnt < self.total_time * self.frequency:
            self.cpg.update(self.mode, self.swing_time)
            self.Position += np.array([[avg_vel * self.dt], [0], [0]])
            for i in range(4):
                self.legs[i].updateDynamicState(self.dt)
                if self.cpg.contact[i] == False:
                    # Stance Phase
                    self.legs[i].moveStance(avg_vel)
                else:
                    # Flight Phase
                    if self.cpg.contact != prev_contact:
                        # Plan lift off trajectory
                        t_sw = self.dt
                        print("Lift ", i, "\n---")
                        swing_config = self.generateSwingTrajectory(i)
                        avg_vel = self.getAverageVelocity(swing_config[1], i)[0, 0]
                        prev_contact = self.cpg.contact.copy()

                    sp = self.getSwingPoint(
                        t_sw, swing_config[0], swing_config[2], swing_config[3]
                    )
                    bf_sp = sp - self.Position - self.legs_offset[i]
                    if i == 0 or i == 3:
                        lf_sp = np.array([[-1, 0, 0], [0, 0, 1]]) @ bf_sp
                    else:
                        lf_sp = np.array([[1, 0, 0], [0, 0, 1]]) @ bf_sp

                    F_fl, T_fl = self.legs[i].getSwingLegInvDynamics()
                    T_ffl = np.cross(self.legs_offset[i].T, F_fl.T).T
                    self.legs[i].moveSwing(lf_sp)

            Fr_ = np.array([[0], [0], [-9.81 * self.m_body]])
            Nr_ = np.array([[0], [0], [0]])
            s2 = self.evaluateFAStability(Fr_, Nr_)
            # print("s1", self.evaluateFAStability(Fr_, Nr_))

            Fr_ = np.array([[0], [0], [-9.81 * self.m_body]]) + F_fl
            # print("F_fl", F_fl)
            Nr_ = np.array([[0], [0], [0]]) + T_fl + T_ffl
            s = self.evaluateFAStability(Fr_, Nr_)
            # print("s2", s)
            print("Diff between Fight leg consider", s2 - s)

            self.performances.append(s)
            self.t_list.append(self.iter_t)

            self.recordTrajectory()
            self.iter_t += self.dt
            t_sw += self.dt
            self.loop_cnt += 1

        # plt.plot(self.t_list, self.performance)
        # plt.show()

    def getHipPosition(self, idx, com_pose):
        return com_pose + self.legs_offset[idx]

    def generateSwingTrajectory(self, leg_idx):
        leg = self.legs[leg_idx]

        if leg_idx == 0 or leg_idx == 3:
            v_OG = np.array([[-1, 0], [0, 0], [0, 1]]) @ leg.vec_OG
        else:
            v_OG = np.array([[1, 0], [0, 0], [0, 1]]) @ leg.vec_OG

        leg_cp1 = self.Position + self.legs_offset[leg_idx] + v_OG
        leg_cp2 = leg_cp1 + np.array([[self.step_length], [0], [0]])

        """ print("leg_cp1, self.Position.T, v_OG.T:",
              leg_cp1.T, self.Position.T, v_OG.T) """

        # quadratic trajectory y(t) = cff*t(t-T)
        cff_x = self.step_length / self.swing_time
        cff_y = -2 * self.step_height / (self.swing_time**2)
        return [leg_cp1, leg_cp2, cff_x, cff_y]

    def getSwingPoint(self, t, cp1, cff_x, cff_y):
        # Return Foothold point of swing foot in world frame
        dx = cff_x * t
        dz = cff_y * t * (t - self.swing_time)
        return cp1 + np.array([[dx], [0], [dz]])

    def getAverageVelocity(self, cp2, lift_idx):
        cps1 = np.array([[0], [0], [0]])
        cps2 = np.array([[0], [0], [0]])

        for i in range(4):
            if i == 0 or i == 3:
                v_OG = np.array([[-1, 0], [0, 0], [0, 1]]) @ self.legs[i].vec_OG
            else:
                v_OG = np.array([[1, 0], [0, 0], [0, 1]]) @ self.legs[i].vec_OG

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
                angle_ = sign * np.arccos(
                    (F_.T @ l_) / (np.linalg.norm(F_) * np.linalg.norm(l_))
                )
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
            coms_ += self.m_leg * (
                self.Position
                + self.legs_offset[i]
                + (rm_ / np.linalg.norm(v_OG)) * v_OG
            )
        coms_ += self.m_body * self.Position
        coms_ = coms_ / 4 * self.m
        return coms_

    def recordTrajectory(self):
        leg_state = [self.iter_t, self.Position.reshape(1, -1).ravel().tolist()]
        for leg in self.legs:
            leg_state.append(leg.state_tb.reshape(1, 2).ravel().tolist())
            leg_state.append(leg.state_phi.reshape(1, 2).tolist()[0])

        leg_state.append(copy.copy(self.cpg.contact))
        # print("[rec] ", self.cpg.contact)
        self.Trajectory.append(leg_state)
        # leg_state = [t, [px, py, pz], [MA_t, MA_b], [MA_phiR, MA_phiL],
        #                               [MB_t, MB_b], [MB_phiR, MB_phiL],
        #                               [MC_t, MC_b], [MC_phiR, MC_phiL],
        #                               [MD_t, MD_b], [MD_phiR, MD_phiL],
        #                               [MOD_A_contact, MOD_B_contact, MOD_C_contact, MOD_D_contact]]
        pass

    def updateAnimation(self, frame_cnt):
        idx = round(frame_cnt * (loop_freq / self.animate_fps))

        if idx <= len(self.Trajectory):
            p_com = self.Trajectory[idx][1]
            p_com_vec = np.array([[p_com[0]], [p_com[1]], [p_com[2]]])
            self.vs_com.set_data(p_com[0], p_com[1])
            self.vs_com.set_3d_properties(p_com[2])

            self.vs_stab_line_t.append(self.Trajectory[idx][0])
            self.vs_stab_line_s.append(self.performances[idx])
            self.vs_stab.set_data(self.vs_stab_line_t, self.vs_stab_line_s)

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
                # lf_OG = vec_OG(phiR_, phiL_)
                lf_OG = lk.FowardKinematics(np.array([[phiR_], [phiL_]]), "phi")
                if i == 0 or i == 3:
                    v_OG = np.array([[-1, 0], [0, 0], [0, 1]]) @ lf_OG
                else:
                    v_OG = np.array([[1, 0], [0, 0], [0, 1]]) @ lf_OG
                p_G = p_hip[i] + v_OG

                self.vs_fhs[i].set_data(
                    [p_hip[i][0, 0], p_G[0, 0]], [p_hip[i][1, 0], p_G[1, 0]]
                )
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

        print(self.vs_stab_line_s)

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
        self.ax1.axes.set_xlim3d([-0.25, 1.75])
        self.ax1.axes.set_ylim3d([-1, 1])
        self.ax1.axes.set_zlim3d([0, 0.8])

        self.vs_com = self.ax1.plot([], [], [], "o", color=color_body[1])[0]
        for i in range(4):
            self.vs_fhs.append(
                self.ax1.plot([], [], [], "o-", color=color_modlist[i])[0]
            )
        self.vs_sup_polygon = self.ax1.plot([], [], [], "o-", color=color_body[1])[0]
        self.vs_com_proj = self.ax1.plot([], [], [], "D", color="orange")[0]

        self.ax2 = fig.add_subplot(2, 1, 2)
        self.ax2.axes.set_xlim([0, self.Trajectory[-1][0]])
        self.ax2.axes.set_ylim([-6, 6])
        self.ax2.axes.set_xlabel("time [sec]")
        self.ax2.axes.set_ylabel("Force Angle Stability")
        self.vs_stab = self.ax2.plot([], [], "o-")[0]

        ani = animation.FuncAnimation(
            fig,
            self.updateAnimation,
            frames=frame_count,
            interval=frame_interval,
            repeat=True,
            init_func=self.resetVisualize(),
        )

        # ani.save("animation.mp4")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    print("CPG Started")

    loop_freq = 1000  # Hz
    FSM = FiniteStateMachine(loop_freq)
    CORGI = Corgi(loop_freq, FSM)
    LDM = sm.SimplifiedModel(data_analysis=False)

    CORGI.standUp(0.05)
    CORGI.move()
    CORGI.visualize()
