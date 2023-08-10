import numpy as np
import linkleg_transform as lt
import dill
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


MODE_STOP = 0
MODE_STANCE = 1
MODE_WALK = 2
MODE_TROT = 3

color_modlist = ["cornflowerblue", "forestgreen", "darkorange", "plum"]
color_body = ["darkred", "indianred"]

with open('./serialized_object/vec_OG_NP.pkl', 'rb') as d:
    vec_OG = dill.load(d)


class FiniteStateMachine:
    def __init__(self, loop_freq):
        # list order -> Mod_A, Mod_B, Mod_C, Mod_D
        # True -> Swing, False -> Stance
        self.contact = [False, False, False, False]
        self.swing_counts = [0, 0, 0, 0]
        self.trigger_round = [1, 3, 2, 4]
        self.count = 0

        self.timer_trigger_time = 0
        self.timer_count = 0
        self.frequency = loop_freq

    def update(self, mode, swing_time):
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
        self.vec_OG = vec_OG(self.state_phi[0, 0], self.state_phi[1, 0])
        self.len_OG = np.linalg.norm(self.vec_OG)

        # quadratic Trajectory coeff
        self.swingTrajectory = [0, 0, 0]

    def updateState(self, tb):
        self.state_tb = tb
        self.state_phi = lt.getPhiRL(tb)
        # self.vec_OG = vec_OG(self.state_phi[0], self.state_phi[1])
        # self.len_OG = np.linalg.norm(self.vec_OG)

    def moveStance(self, vel):
        if self.idx == 0 or self.idx == 3:
            vel *= -1
        dx = vel/self.freq
        vec_dx = np.array([[dx], [0]])
        vec_OG_k = self.vec_OG - vec_dx
        leglen_k = np.linalg.norm(vec_OG_k)

        # get theta_k from inverse kinematics approx...
        theta_k = self.solveLegIK(leglen_k, self.state_tb)

        dbeta_k = np.arccos((vec_OG_k.T@self.vec_OG) /
                            (self.len_OG*leglen_k))[0, 0]

        # Update
        self.len_OG = leglen_k
        self.vec_OG = vec_OG_k
        beta_k = self.state_tb[1, 0] + dbeta_k
        self.state_tb = np.array([[theta_k], [beta_k]])
        self.state_phi = lt.getPhiRL(self.state_tb)

    def genLiftoffTrajectory(self, swing_time, step_length, step_height):
        vec_OG_landing = self.vec_OG + np.array([[step_length], [0]])
        vec_traj_top = self.vec_OG + np.array([[step_length/2], [step_height]])

        C = np.array([[0, 0, 1],
                      [1/2 * swing_time**2, 1/2 * swing_time, 1],
                      [swing_time**2, swing_time, 1]])

        quad_coeff = np.linalg.inv(C) \
            @ np.array([[self.vec_OG[1, 0]], [vec_traj_top[1, 0]], [vec_OG_landing[1, 0]]])

        # quadratic eqn. y = c2*t^2 + c1*t + c0
        c2 = quad_coeff[0, 0]
        c1 = quad_coeff[1, 0]
        c0 = quad_coeff[2, 0]
        self.swingTrajectory = [c2, c1, c0]

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
        beta *= beta_sign

        # print("swing ", np.degrees(theta), np.degrees(beta))
        self.updateState(np.array([[theta], [beta]]))
        self.vec_OG = sp
        self.len_OG = l_des

    def solveLegIK(self, desired_length, current_tb):
        # Solve theta correspond to desired length
        if desired_length > 0.3428:
            return "error"
        else:
            phiRL = lt.getPhiRL(current_tb)
            v_OG = vec_OG(phiRL[0, 0], phiRL[1, 0])
            iter_length = np.sqrt(v_OG.T @ v_OG)[0, 0]
            theta = current_tb[0, 0]

            error = desired_length - iter_length

            tolerance = 0.001
            p_gain = 10

            while error > tolerance:
                theta = theta + (p_gain * error)
                phiRL = lt.getPhiRL(np.array([[theta], [0]]))
                v_OG = vec_OG(phiRL[0, 0], phiRL[1, 0])
                iter_length = np.sqrt(v_OG.T @ v_OG)[0, 0]
                error = desired_length - iter_length
            return theta


class Corgi:
    def __init__(self, loop_freq, cpg):
        self.cpg = cpg
        self.mode = MODE_WALK
        self.step_length = 0.2
        self.step_height = 0.1
        # swing_time = step_length/heading_velocity
        self.swing_time = 1
        self.stance_height = 0.2

        # Robot initial state (in Theta-Beta Representation)
        self.Position = np.array([[0], [0], [0.1]])
        self.Orientation = np.array([0, 0, 0, 1])  # xyzw

        # Physics Prop. (SI unit)
        self.g = np.array([[0], [0], [-9.80665]])
        self.m = 26.662009  # kg
        self.d_l = 577.5 * 0.001
        self.d_w = 329.5 * 0.001
        self.d_h = 144 * 0.001
        self.d_shaft = 0.444
        self.I_zz = 1/12 * self.m * (self.d_l**2 + self.d_w**2)
        self.I_xx = 1/12 * self.m * (self.d_h**2 + self.d_w**2)
        self.I_yy = 1/12 * self.m * (self.d_h**2 + self.d_l**2)
        self.I = np.array([[self.I_xx, 0, 0],
                           [0, self.I_yy, 0],
                           [0, 0, self.I_zz]])

        # Module initial state (in Theta-Beta Representation)
        self.leg_A = LinkLeg(0, loop_freq, np.array([[np.deg2rad(17)], [0]]))
        self.leg_B = LinkLeg(1, loop_freq, np.array([[np.deg2rad(17)], [0]]))
        self.leg_C = LinkLeg(2, loop_freq, np.array([[np.deg2rad(17)], [0]]))
        self.leg_D = LinkLeg(3, loop_freq, np.array([[np.deg2rad(17)], [0]]))
        self.vec_C_AO = np.array([[self.d_shaft/2], [self.d_w/2], [0]])
        self.vec_C_BO = np.array([[self.d_shaft/2], [-self.d_w/2], [0]])
        self.vec_C_CO = np.array([[-self.d_shaft/2], [self.d_w/2], [0]])
        self.vec_C_DO = np.array([[-self.d_shaft/2], [-self.d_w/2], [0]])

        self.legs = [self.leg_A, self.leg_B, self.leg_C, self.leg_D]
        self.legs_offset = [self.vec_C_AO,
                            self.vec_C_BO,
                            self.vec_C_CO,
                            self.vec_C_DO]

        self.frequency = loop_freq
        self.dt = 1/loop_freq

        self.iter_t = 0
        self.loop_cnt = 0
        self.swing_cff_x = 0
        self.swing_cff_y = 0
        self.trajectory = []

    def standUp(self, vel):
        print("Standing Up ...")
        tolerance = 0.01
        t = time.time()
        while np.abs(self.stance_height - self.Position[2, 0]) > tolerance:
            for leg in self.legs:
                theta_ = leg.solveLegIK(leg.len_OG + vel*self.dt, leg.state_tb)
                leg.updateState(np.array([[theta_], [0]]))
                leg.len_OG += vel*self.dt
                leg.vec_OG[1, 0] -= vel*self.dt

            self.Position = self.Position + np.array([[0], [0], [vel*self.dt]])
            self.recordTrajectory()
            self.iter_t += self.dt

        print("Time Elapsed = ", time.time() - t)
        print("iter_t = ", self.iter_t)

    def move(self):
        liftoff_foot = -1
        avg_vel = 0
        t_sw = 0
        sw_ = []
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        while self.loop_cnt < 40 * self.frequency:
            self.cpg.update(self.mode, self.swing_time)

            self.Position += np.array([[avg_vel*self.dt], [0], [0]])
            ax.scatter(self.Position[0, 0],
                       self.Position[1, 0],
                       self.Position[2, 0], color=color_body[0])

            for i in range(4):
                if self.cpg.contact[i] == False:
                    # Stance Phase
                    self.legs[i].moveStance(avg_vel)
                    """ if i == 0:
                        print("[stance] leg:", i, "state_tb:",
                              np.rad2deg(self.legs[i].state_tb.T)) """
                else:
                    # Flight Phase
                    if i != liftoff_foot:
                        t_sw = self.dt
                        # Plan lift off trajectory
                        print("Lift ", i)
                        sw_ = self.generateSwingTrajectory(i)
                        avg_vel = self.getAverageVelocity(sw_[1], i)[0, 0]
                        liftoff_foot = i
                        print("---------")

                    sp = self.getSwingPoint(t_sw, sw_[0], sw_[2], sw_[3])

                    ax.scatter(sp[0, 0], sp[1, 0], sp[2, 0],
                               color=color_modlist[i])

                    bf_sp = sp - self.Position - self.legs_offset[i]
                    if i == 0 or i == 3:
                        lf_sp = np.array([[-1, 0, 0], [0, 0, 1]]) @ bf_sp
                    else:
                        lf_sp = np.array([[1, 0, 0], [0, 0, 1]]) @ bf_sp

                    self.legs[i].moveSwing(lf_sp)
                    """ if i == 0:
                        print("[swing] t_sw:", t_sw)
                        print("[swing] sp:", sp.T)
                        print("[swing] sw_:", sw_)
                        print("[swing] leg:", i, "state_tb:",
                              np.rad2deg(self.legs[i].state_tb.T))
                        print("lf_sp:", lf_sp.T)
                        print("swing_count, contact",
                              self.cpg.swing_counts, self.cpg.contact)
                        print("===") """

                """ p_hip = self.getHipPosition(i)
                ax.scatter(p_hip[0, 0], p_hip[1, 0],
                           p_hip[2, 0], color=color_body[1]) """

            self.recordTrajectory()
            self.iter_t += self.dt
            t_sw += self.dt
            self.loop_cnt += 1
        # print(self.trajectory)
        plt.show()

    def getHipPosition(self, idx):
        return self.Position + self.legs_offset[idx]

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
        cff_x = self.step_length/self.swing_time
        cff_y = -2*self.step_height/(self.swing_time**2)
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
                v_OG = np.array([[-1, 0], [0, 0], [0, 1]]
                                ) @ self.legs[i].vec_OG
            else:
                v_OG = np.array([[1, 0], [0, 0], [0, 1]]) @ self.legs[i].vec_OG

            cp = self.Position + self.legs_offset[i] + v_OG
            if i != lift_idx:
                cps1 = cps1 + cp
                cps2 = cps2 + cp
            else:
                cps1 = cps1 + cp
                cps2 = cps2 + cp2

        com1 = 1/4 * cps1
        com2 = 1/4 * cps2
        vel = (com2 - com1)/self.swing_time
        return vel

    def recordTrajectory(self):
        leg_state = [self.iter_t, self.Position.reshape(
            1, -1).ravel().tolist()]
        for leg in self.legs:
            leg_state.append(leg.state_tb.reshape(1, 2).ravel().tolist())
            leg_state.append(leg.state_phi.reshape(1, 2).tolist()[0])
        self.trajectory.append(leg_state)
        pass

    # def updateAnimation(self):

    def visualize(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")


if __name__ == '__main__':
    print("CPG Started")

    loop_cnt = 0
    loop_freq = 1000  # Hz
    FSM = FiniteStateMachine(loop_freq)
    CORGI = Corgi(loop_freq, FSM)

    CORGI.standUp(0.05)
    CORGI.move()
