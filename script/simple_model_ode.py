import argparse
import math
import numpy as np
import pandas as pd
from scipy.integrate import odeint, solve_ivp
from scipy import interpolate

import linkleg_transform as lt
import matplotlib.pyplot as plt


def get_parser():
    parser = argparse.ArgumentParser(
        description='Simplified Model Description')
    parser.add_argument('-f', '--datapath',
                        default='../sbrio_data/loadcell/loadcell_data_0519/20230519_sinewave_t_90_45_2_b_0_0_1_22.csv')
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
        self.importData(filepath)
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


class SimplifiedModel:
    def __init__(self, filepath) -> None:
        # Dynamic Model Properties
        self.m = 0.654  # Total Mass of link leg
        self.g = 9.81  # Gravity
        self.Fc = 1.4  # Columb Friction Coeff.
        self.Fv = 0.72  # Viscous Friction Coeff.

        self.data = SBRIO_data(filepath)

        # ode setup
        self.tspan = (self.data.t[0, 0], self.data.t[-1, 0])
        self.tau = self.data.rpy_trq_phi

        # set initial condition from data
        init_tb = lt.getThetaBeta(np.reshape(
            self.data.rpy_pos_phi[0, :], (2, 1)))
        init_Rm = lt.getRm(init_tb[0, 0])
        self.init_condition = [init_Rm, 0, init_tb[1, 0], 0]

        result_solve_ivp = solve_ivp(
            self.linkLegODE, self.tspan, self.init_condition, atol=1e-2, rtol=1e-2)

        rpy_rm = []
        for i in range(self.data.rpy_pos_phi.shape[0]):
            tb_ = lt.getThetaBeta(self.data.rpy_pos_phi[i, :].reshape((2, 1)))
            rpy_rm.append(lt.getRm(tb_[0, 0]))

        print("-- ODE45 Solver Result --")
        print(result_solve_ivp)
        plt.plot(result_solve_ivp.t, result_solve_ivp.y[0, :])
        plt.plot(self.data.t, rpy_rm)
        plt.show()
        pass

    def linkLegODE(self, t, State):
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
            print("Theta Polynomial Root Solution Not Found !")
            print("t =", t)
            print("root =", theta_roots)
            print("---")
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
            J_r = np.mat([[4*rm_coeff[4]*theta_**3 + 3*rm_coeff[3]*theta_**2 + 2*rm_coeff[2]*theta_ + rm_coeff[1], 0],
                          [0, 1]])
            d_phi_RL = np.linalg.inv(J_r * J1) * np.mat([[d_rm_], [d_beta_]])

            tau_friction_R = - \
                (self.Fc * np.sign(d_phi_RL[0, 0]) + self.Fv * d_phi_RL[0, 0])
            tau_friction_L = - \
                (self.Fc * np.sign(d_phi_RL[1, 0]) + self.Fv * d_phi_RL[1, 0])
            tau_total = np.array([[tau_phi_R + tau_friction_R],
                                  [tau_phi_L + tau_friction_L]])

            Frm_, Tb_ = lt.getFrmTb(tau_total, theta_)

            # get d_Ic from [d_rm_; d_beta];
            J_Ic = np.mat([[4*Icom_coeff[4]*theta_**3 + 3*Icom_coeff[3]*theta_**2 + 2*Icom_coeff[2]*theta_ + Icom_coeff[1], 0],
                           [0, 1]])
            d_Ic = J_Ic * np.linalg.inv(J_r) * np.mat([[d_rm_], [d_beta_]])
            d_Ic = d_Ic[0, 0]

            I_com = lt.getIc(theta_)
            I_hip = I_com + self.m * rm_**2

            # Obtain dd_rm_, dd_beta_ from Lagrange equation
            dd_rm_ = (Frm_ + self.m*rm_*d_beta_**2 +
                      self.m*self.g*np.cos(beta_))/self.m

            dd_beta_ = (1/I_hip) * (Tb_ - 2*self.m*rm_*d_rm_*d_beta_ -
                                    d_Ic*d_beta_ - self.m*self.g*rm_*np.sin(beta_))

            return [d_rm_, dd_rm_[0, 0], d_beta_, dd_beta_[0, 0]]


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    print("--- Simplified link leg Model ---")
    print('Data: ' + args.datapath)

    # sb_ = SBRIO_data(args.datapath)
    # sb_.importData(args.datapath)
    sm_ = SimplifiedModel(args.datapath)

    pass
