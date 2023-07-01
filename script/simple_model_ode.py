import argparse
import math
import numpy as np
import pandas as pd
from scipy.integrate import odeint, solve_ivp
from scipy import interpolate
from scipy import signal

import linkleg_transform as lt
import matplotlib.pyplot as plt


def get_parser():
    parser = argparse.ArgumentParser(
        description='Simplified Model Description')
    parser.add_argument('-f', '--datapath',
                        default='sbrio_data/loadcell/loadcell_data_0519/20230519_sinewave_t_90_45_5_b_0_0_1_24.csv')
    return parser


class SBRIO_data:
    def __init__(self, filepath) -> None:
        self.KT_comp = 2.2
        self.t = []
        self.rpy_pos_phi = []
        self.rpy_vel_phi = []
        self.rpy_trq_phi = []
        self.cmd_pos_phi = []
        self.loadcell = []

        self.rpy_pos_tb = []
        self.rpy_vel_tb = []
        self.rpy_acc_tb = []

        self.invdyn_tauFrmTb_total = []
        self.invdyn_tauRL_total = []
        self.importData(filepath)
        self.diffData()
        pass

    def importData(self, filepath):
        # import csv file from SBRIO
        df = pd.read_csv(filepath)
        raw_dfshape = df.shape

        print("Imported Data shape :", df.shape)

        init_row = 0
        for i in range(df.shape[0]):
            if df.iloc[i+1, 1] != 0:
                init_row = i
                break

        print("First Row :", init_row)
        df = df.iloc[init_row:(raw_dfshape[0]-1), :]

        last_row = 0
        for i in range(df.shape[0]):
            if df.iloc[i, 0] == 0:
                last_row = i
                break

        df = df.iloc[0:last_row, :]
        print("Last row :", init_row + last_row)
        print("Trimmed Data shape :", df.shape)
        # print(list(df.iloc[0, :]))
        rpy_loadcell = (df.iloc[:, 58]-0.654) * 9.80665
        self.loadcell = np.array(rpy_loadcell).reshape((-1, 1))

        rpy_pos_phi_R = df.iloc[:, 13]
        rpy_pos_phi_L = df.iloc[:, 16]
        self.rpy_pos_phi = np.array([rpy_pos_phi_R, rpy_pos_phi_L]).T

        rpy_vel_phi_R = df.iloc[:, 14]
        rpy_vel_phi_L = df.iloc[:, 17]
        self.rpy_vel_phi = np.array([rpy_vel_phi_R, rpy_vel_phi_L]).T

        rpy_trq_phi_R = df.iloc[:, 15]
        rpy_trq_phi_L = df.iloc[:, 18]
        self.rpy_trq_phi = self.KT_comp * \
            np.array([rpy_trq_phi_R, rpy_trq_phi_L]).T

        cmd_pos_phi_R = df.iloc[:, 1]
        cmd_pos_phi_L = df.iloc[:, 6]
        self.cmd_pos_phi = np.array([cmd_pos_phi_R, cmd_pos_phi_L]).T

        t0_us = df.iloc[0, 0]
        self.t = (np.array([df.iloc[:, 0]]) - t0_us).T * 1e-6
        pass

    def diffData(self):
        for phiRL in self.rpy_pos_phi:
            self.rpy_pos_tb.append(lt.getThetaBeta(phiRL[None].T).T)
        self.rpy_pos_tb = np.array(self.rpy_pos_tb).reshape((-1, 2))

        for velRL in self.rpy_vel_phi:
            J_tb = np.matrix([[1/2, -1/2], [1/2, 1/2]])
            vel_tb = J_tb * velRL[None].T
            self.rpy_vel_tb.append(vel_tb.T)
        self.rpy_vel_tb = np.array(self.rpy_vel_tb).reshape((-1, 2))

        self.rpy_vel_tb[:, 0] = signal.medfilt(self.rpy_vel_tb[:, 0], 9)
        self.rpy_vel_tb[:, 1] = signal.medfilt(self.rpy_vel_tb[:, 1], 9)

        self.rpy_acc_tb = (np.diff(self.rpy_vel_tb.T) / np.diff(self.t.T)).T
        self.rpy_acc_tb[:, 0] = signal.medfilt(self.rpy_acc_tb[:, 0], 11)
        self.rpy_acc_tb[:, 1] = signal.medfilt(self.rpy_acc_tb[:, 1], 11)


class SimplifiedModel:
    def __init__(self, filepath) -> None:
        # Dynamic Model Properties
        self.m = 0.654  # Total Mass of link leg
        self.g = 9.81  # Gravity
        self.Fc = 1.4  # Columb Friction Coeff.
        self.Fv = 0.72  # Viscous Friction Coeff.

        self.data = SBRIO_data("../"+filepath)
        # self.data = filepath

        # foward dynamic ode setup
        self.tspan = (self.data.t[0, 0], self.data.t[-1, 0])
        self.tau = self.data.rpy_trq_phi

        # set initial condition from data
        init_tb = lt.getThetaBeta(np.reshape(
            self.data.rpy_pos_phi[0, :], (2, 1)))
        init_Rm = lt.getRm(init_tb[0, 0])
        self.init_condition = [init_Rm, 0, init_tb[1, 0], 0]

        # iterate foward dynamic model
        self.iterate_freq = 100  # Hz
        self.iterate_dt = 1/self.iterate_freq
        self.iterate_horizon = 0.025  # second
        self.iterate_trajectory = []

    def modelIteration(self):
        # for i in range(self.data.t.shape[0]):
        #     print(self.data.t[i])
        #     # print(i)
        iter_t0 = 0
        while iter_t0 < self.data.t[-1]:
            # get initial condition
            print("--- ", iter_t0, " ----")
            phi_R_ = np.interp(iter_t0, self.data.t.T.ravel(),
                               self.data.rpy_pos_phi[:, 0].T)
            phi_L_ = np.interp(iter_t0, self.data.t.T.ravel(),
                               self.data.rpy_pos_phi[:, 1].T)

            d_phi_R_ = np.interp(iter_t0, self.data.t.T.ravel(),
                                 self.data.rpy_vel_phi[:, 0].T)
            d_phi_L_ = np.interp(iter_t0, self.data.t.T.ravel(),
                                 self.data.rpy_vel_phi[:, 1].T)

            tb_ = lt.getThetaBeta(np.array([[phi_R_], [phi_L_]]))
            theta_ = tb_[0, 0]
            beta_ = tb_[1, 0]
            rm_ = lt.getRm(theta_)

            J1 = np.mat([[1/2, -1/2],
                        [1/2, 1/2]])

            Jr = np.mat([[4*lt.rm_coeff[4]*theta_**3 + 3*lt.rm_coeff[3]*theta_**2 + 2*lt.rm_coeff[2]*theta_ + lt.rm_coeff[1], 0],
                         [0, 1]])

            d_rb_ = Jr * J1 * np.mat([[d_phi_R_], [d_phi_L_]])
            d_rm_ = d_rb_[0, 0]
            d_beta_ = d_rb_[1, 0]

            init_condition = [rm_, d_rm_, beta_, d_beta_]
            tspan = [iter_t0, iter_t0 + self.iterate_horizon]

            # Numerical Solution ODE45
            # result_solve_ivp = solve_ivp(
            #     self.fowardLinkLegODE, self.tspan, self.init_condition, atol=1e-2, rtol=1e-2)

            # Numerical Solution Foward Euler Method
            traj = self.fowardEulerMethod(self.fowardLinkLegODE,
                                          init_condition, tspan, 0.0025)
            plt.plot(traj[:, 0], traj[:, 1], alpha=0.4)
            iter_t0 += self.iterate_dt
            pass

        rpy_rm = []
        for i in range(self.data.rpy_pos_phi.shape[0]):
            tb_ = lt.getThetaBeta(self.data.rpy_pos_phi[i, :].reshape((2, 1)))
            rpy_rm.append(lt.getRm(tb_[0, 0]))
        plt.plot(self.data.t, rpy_rm, linestyle="--")
        plt.xlim([7.2, 7.5])
        plt.ylim([0.025, 0.16])
        plt.grid()
        plt.show()

        pass

    def fowardEulerMethod(self, ode, ic, tspan, step_size):
        t_ = tspan[0]
        state_ = ic
        traj = np.append(t_, state_)
        while (t_ <= tspan[1]):
            t_ += step_size
            state_ += np.array(ode(t_, state_)) * step_size
            traj = np.append(traj, t_)
            traj = np.append(traj, state_)
        traj = traj.reshape(-1, 5)
        return traj

    def fowardLinkLegODE(self, t, State):
        rm_, d_rm_, beta_, d_beta_ = State

        rm_coeff = [-0.0132, 0.0500, 0.0030, 0.0110, -0.0035]
        Icom_coeff = [0.0041, 0.0043, -0.0013, -0.0001, 0.0001]

        # Get current theta from rm_
        # High order to low order
        rm_coeff_func_flip = [-0.0035, 0.0110, 0.0030, 0.0500, -0.0132-rm_]
        theta_roots = np.roots(rm_coeff_func_flip)
        theta_ = 0
        theta_found = False
        for root in theta_roots:
            if root.imag == 0.:
                if root.real >= np.deg2rad(16.9) and root.real <= np.deg2rad(160.1):
                    theta_ = root.real
                    theta_found = True

        if theta_found == False:
            # print("Theta Polynomial Root Solution Not Found !")
            # print("t =", t)
            # print("root =", theta_roots)
            # print("---")
            return False
        else:
            # Transform Torque input to F_rm and T_beta
            tau_phi_R_func = interpolate.interp1d(
                self.data.t.T.ravel(), self.tau[:, 0].T, fill_value='extrapolate')
            tau_phi_L_func = interpolate.interp1d(
                self.data.t.T.ravel(), self.tau[:, 1].T, fill_value='extrapolate')
            tau_phi_R = tau_phi_R_func(t)
            tau_phi_L = tau_phi_L_func(t)
            # tau_phi_R = np.interp(t, self.data.t.T.ravel(), self.tau[:, 0].T)
            # tau_phi_L = np.interp(t, self.data.t.T.ravel(), self.tau[:, 1].T)

            # Take Joint Friction Force into account
            J1 = np.mat([[1/2, -1/2],
                        [1/2, 1/2]])
            Jr = np.mat([[4*rm_coeff[4]*theta_**3 + 3*rm_coeff[3]*theta_**2 + 2*rm_coeff[2]*theta_ + rm_coeff[1], 0],
                         [0, 1]])
            d_phi_RL = np.linalg.inv(Jr * J1) * np.mat([[d_rm_], [d_beta_]])

            '''
            tau_friction_R = - \
                (self.Fc * np.sign(d_phi_RL[0, 0]) + self.Fv * d_phi_RL[0, 0])
            tau_friction_L = - \
                (self.Fc * np.sign(d_phi_RL[1, 0]) + self.Fv * d_phi_RL[1, 0])
            '''

            # Ichia Version Friction model
            tau_friction_R = - \
                (0.45*np.sign(d_phi_RL[0, 0]) + 0.28 *
                 np.sign(d_phi_RL[0, 0])*tau_phi_R)
            tau_friction_L = - \
                (0.45*np.sign(d_phi_RL[1, 0]) + 0.28 *
                 np.sign(d_phi_RL[1, 0])*tau_phi_R)
            # tau_friction_R = 0
            # tau_friction_L = 0

            tau_total = np.array([[tau_phi_R + tau_friction_R],
                                  [tau_phi_L + tau_friction_L]])

            Frm_, Tb_ = lt.getFrmTb(tau_total, theta_)

            # get d_Ic from [d_rm_; d_beta];
            J_Ic = np.mat([[4*Icom_coeff[4]*theta_**3 + 3*Icom_coeff[3]*theta_**2 + 2*Icom_coeff[2]*theta_ + Icom_coeff[1], 0],
                           [0, 1]])
            d_Ic = J_Ic * np.linalg.inv(Jr) * np.mat([[d_rm_], [d_beta_]])
            d_Ic = d_Ic[0, 0]

            I_com = lt.getIc(theta_)
            I_hip = I_com + self.m * rm_**2

            # Obtain dd_rm_, dd_beta_ from Lagrange equation
            dd_rm_ = (Frm_ + self.m*rm_*d_beta_**2 +
                      self.m*self.g*np.cos(beta_))/self.m

            dd_beta_ = (1/I_hip) * (Tb_ - 2*self.m*rm_*d_rm_*d_beta_ -
                                    d_Ic*d_beta_ - self.m*self.g*rm_*np.sin(beta_))

            return [d_rm_, dd_rm_[0, 0], d_beta_, dd_beta_[0, 0]]

    def solveFowardDyanamic(self):
        result_solve_ivp = solve_ivp(
            self.fowardLinkLegODE, self.tspan, self.init_condition, atol=1e-2, rtol=1e-2)

        rpy_rm = []
        for i in range(self.data.rpy_pos_phi.shape[0]):
            tb_ = lt.getThetaBeta(self.data.rpy_pos_phi[i, :].reshape((2, 1)))
            rpy_rm.append(lt.getRm(tb_[0, 0]))

        print("-- ODE45 Solver Result --")
        print(result_solve_ivp)
        plt.plot(result_solve_ivp.t, result_solve_ivp.y[0, :])
        plt.plot(self.data.t, rpy_rm)
        plt.grid()
        plt.show()

    def inverseLinkLegODE(self, State):
        # State defined as [theta, dtheta, ddtheta, beta, dbeta, ddbeta]
        theta_, dtheta_, ddtheta_, beta_, dbeta_, ddbeta_ = State
        rm_ = lt.getRm(theta_)
        drm_ = lt.get_dRm(theta_, dtheta_)
        ddrm_ = lt.get_ddRm(theta_, dtheta_, ddtheta_)

        Ic_ = lt.getIc(theta_)
        dIc_ = lt.get_dIc(theta_, dtheta_)

        F_rm = self.m*ddrm_ - self.m*rm_ * \
            dbeta_**2 - self.m*self.g*np.cos(beta_)
        T_beta = (Ic_ + self.m*rm_**2)*ddbeta_ + 2*self.m*rm_*drm_ * \
            dbeta_ + dIc_*dbeta_ + self.m*self.g*rm_*np.sin(beta_)

        return np.array([F_rm, T_beta])

    def iterateInverseDynamic(self, sbrio_data):
        for i in range(sbrio_data.rpy_acc_tb.shape[0]):
            theta = sbrio_data.rpy_pos_tb[i, 0]
            beta = sbrio_data.rpy_pos_tb[i, 1]

            dtheta = sbrio_data.rpy_vel_tb[i, 0]
            dbeta = sbrio_data.rpy_vel_tb[i, 1]

            ddtheta = sbrio_data.rpy_acc_tb[i, 0]
            ddbeta = sbrio_data.rpy_acc_tb[i, 1]
            sbrio_data.invdyn_tauFrmTb_total.append(self.inverseLinkLegODE(
                [theta, dtheta, ddtheta, beta, dbeta, ddbeta]))

        sbrio_data.invdyn_tauFrmTb_total = np.array(
            sbrio_data.invdyn_tauFrmTb_total)

        '''
        fig, ax = plt.subplots()
        print(sbrio_data.loadcell)
        print(sbrio_data.t[1:, 0])
        ax.plot(sbrio_data.t[1:, 0], sbrio_data.id_tau,
                label="inverse dynamic model")
        ax.plot(sbrio_data.t, sbrio_data.loadcell, label="loadcell")
        leg = ax.legend(loc="upper right")
        plt.grid()
        plt.show()
        '''

    def getFrictionTau(self, sbrio_data):
        # Get Tau_friction from inverse dynamic
        for i in range(sbrio_data.rpy_acc_tb.shape[0]):
            theta = sbrio_data.rpy_pos_tb[i, 0]
            beta = sbrio_data.rpy_pos_tb[i, 1]
            dtheta = sbrio_data.rpy_vel_tb[i, 0]
            dbeta = sbrio_data.rpy_vel_tb[i, 1]
            ddtheta = sbrio_data.rpy_acc_tb[i, 0]
            ddbeta = sbrio_data.rpy_acc_tb[i, 1]

            tauFrmTb = sbrio_data.invdyn_tauFrmTb_total[i, :][None].T
            tauRL = lt.getTauRL(tauFrmTb, theta).T
            sbrio_data.invdyn_tauRL_total.append(tauRL)

        sbrio_data.invdyn_tauRL_total = np.array(
            sbrio_data.invdyn_tauRL_total).reshape((-1, 2))

        print(sbrio_data.invdyn_tauRL_total.shape)
        print(sbrio_data.rpy_trq_phi.shape)
        motor_tauRL = sbrio_data.rpy_trq_phi[1:, :]
        motor_velRL = sbrio_data.rpy_vel_phi[1:, :]

        motor_tauRL[:, 0] = signal.medfilt(motor_tauRL[:, 0], 9)
        motor_tauRL[:, 1] = signal.medfilt(motor_tauRL[:, 1], 9)

        print(sbrio_data.invdyn_tauRL_total)
        print(motor_tauRL)
        friction_tauRL = np.mat(
            sbrio_data.invdyn_tauRL_total) - np.mat(motor_tauRL)
        print(friction_tauRL)
        print("friction_tau shape", friction_tauRL.shape)
        print("vel_RL shape", sbrio_data.rpy_vel_phi.shape)
        # plt.plot(motor_tauRL[:, 1])
        plt.plot(motor_tauRL[:, 1], friction_tauRL[:, 1],  linewidth=1)
        # plt.plot(motor_velRL[:, 1])

        plt.show()

        # sbrio_data.invdyn_tauFrmTb_total.append(self.inverseLinkLegODE(
        #     [theta, dtheta, ddtheta, beta, dbeta, ddbeta]))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    print("--- Simplified link leg Model ---")
    print('Data: ' + args.datapath)

    # sb_ = SBRIO_data(args.datapath)
    # sb_.importData(args.datapath)
    sm_ = SimplifiedModel(args.datapath)
    sm_.modelIteration()
    # sm_.iterateInverseDynamic(sm_.data)
    # sm_.getFrictionTau(sm_.data)

    pass
