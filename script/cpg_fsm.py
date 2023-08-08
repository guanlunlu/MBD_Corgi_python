import numpy as np
import linkleg_transform as lt
import dill
import time


MODE_STOP = 0
MODE_STANCE = 1
MODE_WALK = 2
MODE_TROT = 3


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

            if self.swing_counts[i] > 0:
                self.contact[i] = True
            else:
                self.contact[i] = False

        self.count += 1


class LinkLeg:
    def __init__(self, freq, tb):
        self.freq = freq
        self.state_tb = tb
        self.state_phi = lt.getPhiRL(tb)
        self.vec_OG = vec_OG(self.state_phi[0], self.state_phi[1])
        self.len_OG = np.linalg.norm(self.vec_OG)

        # quadratic Trajectory coeff
        self.swingTrajectory = [0, 0, 0]

    def updateState(self, tb):
        self.state_tb = tb
        self.state_phi = lt.getPhiRL(tb)
        # self.vec_OG = vec_OG(self.state_phi[0], self.state_phi[1])
        # self.len_OG = np.linalg.norm(self.vec_OG)

    def moveStance(self, vel):
        dx = vel/self.freq
        vec_dx = np.array([[dx], [0]])
        vec_OG_k = self.vec_OG - vec_dx
        leglen_k = np.linalg.norm(vec_OG_k)

        # get theta_k from inverse kinematics approx...
        theta_k = self.solveLegIK(leglen_k, self.state_tb)

        dbeta_k = np.arccos((vec_OG_k.T@self.vec_OG) / (self.len_OG*leglen_k))

        # Update
        self.len_OG = leglen_k
        self.vec_OG = vec_OG_k
        beta_k = self.state_tb[1] + dbeta_k
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

    def moveSwing(self, t_flight):

        pass

    def solveLegIK(self, desired_length, current_tb):
        # Solve theta correspond to desired length
        phiRL = lt.getPhiRL(current_tb)
        v_OG = vec_OG(phiRL[0, 0], phiRL[1, 0])
        iter_length = np.sqrt(v_OG.T @ v_OG)[0, 0]
        theta = current_tb[0, 0]

        error = desired_length - iter_length

        tolerance = 0.0005
        p_gain = 10

        while error > tolerance:
            theta = theta + (p_gain * error)

            phiRL = lt.getPhiRL(np.array([[theta], [0]]))
            v_OG = vec_OG(phiRL[0, 0], phiRL[1, 0])
            iter_length = np.sqrt(v_OG.T @ v_OG)[0, 0]
            error = desired_length - iter_length

        # print("iter_theta = ", theta)
        # print("iter_len = ", iter_len)
        return theta


class Corgi:
    def __init__(self, loop_freq, cpg):
        self.cpg = cpg
        # Robot initial state (in Theta-Beta Representation)
        self.Position = np.array([[0], [0], [0]])
        self.Orientation = np.array([0, 0, 0, 1])  # xyzw

        # Module initial state (in Theta-Beta Representation)
        self.leg_A = LinkLeg(loop_freq, np.array([[0], [0]]))
        self.leg_B = LinkLeg(loop_freq, np.array([[0], [0]]))
        self.leg_C = LinkLeg(loop_freq, np.array([[0], [0]]))
        self.leg_D = LinkLeg(loop_freq, np.array([[0], [0]]))
        self.legs = [self.leg_A, self.leg_B, self.leg_C, self.leg_D]
        self.frequency = loop_freq
        self.dt = 1/loop_freq

        self.trajectory = []

    def standUp(self, stance_height, vel):
        print("Standing Up ...")
        tolerance = 0.01
        t = time.time()
        while np.abs(stance_height - self.Position[2, 0]) > tolerance:
            for leg in self.legs:
                theta_ = leg.solveLegIK(leg.len_OG + vel*self.dt, leg.state_tb)
                leg.updateState(np.array([[theta_], [0]]))
                leg.len_OG += vel*self.dt
                leg.vec_OG[1, 0] += vel*self.dt

            self.Position = self.Position + np.array([[0], [0], [vel*self.dt]])
            self.recordTrajectory()
        print("Time Elapsed = ", time.time() - t)
        print(self.trajectory)

    def move(self):
        pass

    def recordTrajectory(self):
        leg_state = []
        for leg in self.legs:
            leg_state.append(leg.state_tb.reshape(1, 2).ravel().tolist())
            leg_state.append(leg.state_phi.reshape(1, 2).tolist()[0])
        self.trajectory.append(leg_state)
        pass


if __name__ == '__main__':
    print("CPG Started")

    loop_cnt = 0
    loop_freq = 1000  # Hz
    FSM = FiniteStateMachine(loop_freq)
    CORGI = Corgi(loop_freq, FSM)

    mode = MODE_WALK
    step_length = 0.1
    velocity = 0.1
    # swing_time = step_length/heading_velocity
    swing_time = 1
    stance_height = 0.2

    CORGI.standUp(stance_height, 0.05)

    # while loop_cnt < 10*1000:
    #     FSM.update(mode, swing_time)

    #     # FSM.update(MODE_TROT, swing_time*1000)
    #     print("Iter ", loop_cnt, ": ", FSM.swing_counts, FSM.contact)
    #     print(CORGI.move())

    #     for i in range(4):
    #         if FSM.contact[i] == False:
    #             # Stance
    #             # CORGI.legs[i].moveStance(velocity)
    #             pass
    #         else:
    #             # Lift Off
    #             pass

    #     loop_cnt += 1
