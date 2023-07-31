import numpy as np
from scipy.spatial.transform import Rotation as R


class RobotState:
    def __init__(self, pose, linear_velocity, angular_velocity, orientation):
        self.pose = pose
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity
        # Represent in quaternion [x, y, z, w]
        self.orient = orientation
        self.rotationMatrix = np.zeros((3, 3))

        self.state_vector = np.vstack(
            [self.pose, self.linear_velocity, self.angular_velocity, self.orient])

    def getStateVector(self):
        self.state_vector = np.vstack(
            [self.pose, self.linear_velocity, self.angular_velocity, self.orient])
        return self.state_vector

    def getRotationMatrix(self):
        r = R.from_quat(self.orient)
        rm = r.as_matrix()
        return rm


class Corgi:
    def __init__(self, init_pose, init_lvel, init_avel, init_orient) -> None:
        self.state = RobotState(init_pose, init_lvel, init_avel, init_orient)

        # Physics Prop. (SI unit)
        self.g = np.array([[0], [0], [-9.80665]])
        self.m = 26.662009  # kg
        self.d_l = 577.5 * 0.001
        self.d_w = 329.5 * 0.001
        self.d_h = 144 * 0.001
        self.I_zz = 1/12 * self.m * (self.d_l**2 + self.d_w**2)
        self.I_xx = 1/12 * self.m * (self.d_h**2 + self.d_w**2)
        self.I_yy = 1/12 * self.m * (self.d_h**2 + self.d_l**2)
        self.I = np.array([[self.I_xx, 0, 0],
                           [0, self.I_yy, 0],
                           [0, 0, self.I_zz]])

    def updateState(self, dt, u1, r1, u2, r2, u3, r3, u4, r4):
        # u1, u2, u3, u4 -> Ext. Force Represent in Base frame [Mod A, Mod B, Mod C, Mod D]
        # u1 = [u1_x, u1_y, 0]
        # r1, r2, r3, r4 -> lever of force represent in Base frame (From Point Force App. to C.O.M.)
        U = u1 + u2 + u3 + u4
        U_x = U[0, 0]
        U_z = U[1, 0]
        R_k_1 = self.state.getRotationMatrix()

        # Ext. Force convert to world frame
        F_k = R_k_1 @ np.array([[U_x], [0], [U_z]])

        # Translation Motion
        dd_X_k = self.g + F_k / self.m
        d_X_k = self.state.linear_velocity + dt * dd_X_k
        X_k = self.state.pose + dt * d_X_k

        # Rotation Motion
        # Ext. Torque in Base Frame
        u1_ = np.array([[u1[0, 0]], 0, [u1[1, 0]]])
        u2_ = np.array([[u1[0, 0]], 0, [u1[1, 0]]])
        u3_ = np.array([[u1[0, 0]], 0, [u1[1, 0]]])
        u4_ = np.array([[u1[0, 0]], 0, [u1[1, 0]]])
        t1_ = np.cross(r1, u1_)
        t2_ = np.cross(r2, u2_)
        t3_ = np.cross(r3, u3_)
        t4_ = np.cross(r4, u4_)
        t_ext_ = t1_ + t2_ + t3_ + t4_

        # angular velocity
        d_w_k = self.state.angular_velocity + dt * np.linalg.inv(self.I) @ (
            t_ext_ - np.cross(self.state.angular_velocity, self.I@self.state.angular_velocity))
        w_k = self.state.angular_velocity + dt * d_w_k

        # Orientation
        H_k_1 = self.state.orient
        H_k_1_x = H_k_1[0]
        H_k_1_y = H_k_1[1]
        H_k_1_z = H_k_1[2]
        H_k_1_w = H_k_1[3]
        H_prime_k_1 = np.array([[-H_k_1_x, -H_k_1_y, -H_k_1_z],
                                [H_k_1_w, -H_k_1_z, H_k_1_y],
                                [H_k_1_z, H_k_1_w, -H_k_1_x],
                                [-H_k_1_y, H_k_1_x, H_k_1_w]])

        H_k = H_k_1 + 1/2 @ dt @ H_prime_k_1 @ w_k

        self.state = RobotState(X_k, d_X_k, w_k, H_k)

    def quatMutiply(self, qa, qb):
        qa_x = qa[0]
        qa_y = qa[1]
        qa_z = qa[2]
        qa_w = qa[3]
        qb_x = qb[0]
        qb_y = qb[1]
        qb_z = qb[2]
        qb_w = qb[3]
        Ha = np.array([[qa_w, -qa_x, -qa_y, -qa_z],
                       [qa_x, qa_w, -qa_z, qa_y],
                       [qa_y, qa_z, qa_w, -qa_x],
                       [qa_z, -qa_y, qa_x, qa_w]])
        hb_ = np.array([qb_w, qb_x, qb_y, qb_z])
        # wxyz
        sol_ = Ha@hb_
        return np.vstack(sol_[1:3], sol_[0])
