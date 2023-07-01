
import numpy as np

# Curve Fitting Coeff. (low order to high order)
rm_coeff = [-0.0132, 0.0500, 0.0030, 0.0110, -0.0035]
Icom_coeff = [0.0041, 0.0043, -0.0013, -0.0001, 0.0001]


def getThetaBeta(phiRL):
    # column vector [phiR; phiL]
    return 1/2 * np.mat([[1, -1], [1, 1]]) * np.mat(phiRL) + \
        np.mat([[1], [0]]) * np.deg2rad(17)


def getPhiRL(tb):
    return np.mat([[1, 1], [-1, 1]]) * np.mat(tb) - np.mat([[1], [-1]]) * np.deg2rad(17)


def getFrmTb(T_RL, theta):
    # Transform [T_phi_R; T_phi_L] to [F_rm; T_beta]
    J1 = np.mat([[1/2, -1/2],
                 [1/2, 1/2]])

    J2 = np.mat([[4*rm_coeff[4]*theta**3 + 3*rm_coeff[3]*theta**2 + 2*rm_coeff[2]*theta + rm_coeff[1], 0],
                 [0, 1]])

    tau_FmTb = np.linalg.inv(J2.T) * np.linalg.inv(J1.T) * np.mat(T_RL)
    return tau_FmTb


def getTauRL(FmTb, theta):
    # Transform [F_rm; T_beta] to [T_phi_R; T_phi_L]
    J1 = np.mat([[1/2, -1/2],
                 [1/2, 1/2]])

    J2 = np.mat([[4*rm_coeff[4]*theta**3 + 3*rm_coeff[3]*theta**2 + 2*rm_coeff[2]*theta + rm_coeff[1], 0],
                 [0, 1]])

    tau_RL = J1.T * J2.T * np.mat(FmTb)
    return tau_RL


def getRm(theta):
    Rm = rm_coeff[4]*theta**4 + rm_coeff[3]*theta**3 + \
        rm_coeff[2]*theta**2 + rm_coeff[1]*theta + rm_coeff[0]
    return Rm


def getIc(theta):
    Icom = Icom_coeff[4]*theta**4 + Icom_coeff[3]*theta**3 + \
        Icom_coeff[2]*theta**2 + Icom_coeff[1]*theta + Icom_coeff[0]
    return Icom


def get_dRm(theta, dtheta):
    # Rm partial deriviative to theta
    d_rm_ = (4*rm_coeff[4]*theta**3 + 3*rm_coeff[3] *
             theta**2 + 2*rm_coeff[2]*theta + rm_coeff[1]) * dtheta
    return d_rm_


def get_ddRm(theta, dtheta, ddtheta):
    dd_rm_ = (4*rm_coeff[4]*theta**3 + 3*rm_coeff[3]*theta**2 + 2*rm_coeff[2]*theta + rm_coeff[1]) * \
        ddtheta + (12*rm_coeff[4]*theta**2 + 6*rm_coeff[3]
                   * theta + 2*rm_coeff[2]) * dtheta**2
    return dd_rm_


def get_dIc(theta, dtheta):
    d_Ic = (4*Icom_coeff[4]*theta**3 + 3*Icom_coeff[3]*theta **
            2 + 2*Icom_coeff[2]*theta + Icom_coeff[1])*dtheta
    return d_Ic
