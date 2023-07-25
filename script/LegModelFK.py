import math
import numpy as np
import sympy as smp
from sympy import symbols, Matrix, Transpose

import LegSymbol as ls
import dill


class LegModel:
    def __init__(self, legsym) -> None:
        self.t = legsym.t

        self.phi_R = legsym.phi_R
        self.phi_L = legsym.phi_L

        self.theta = legsym.theta
        self.beta = legsym.beta

        self.vec_phi = Matrix([[self.phi_R], [self.phi_L]])

        # geometric parameters (meters)
        self.R = 0.001*100
        self.theta_0 = math.radians(17)
        n_HF_ = 130/180
        n_BC_ = 101/180

        # linkage length
        # side OA
        self.l1_ = 0.8*self.R
        # side BC
        self.l3_ = 2*self.R*np.sin(np.pi*n_BC_/2)
        # side DC
        self.l4_ = 0.88296634 * self.R
        # side AD
        self.l5_ = 0.9*self.R
        # side DE
        self.l6_ = 0.4*self.R
        # side CF
        self.l7_ = 2*self.R*np.sin((130-17-101)*np.pi/180/2)
        # side GF
        self.l8_ = 2*self.R*np.sin(math.radians(25))
        self.l_BF = 0

        self.theta = (1/2 * Matrix([[1, -1], [1, 1]]) * Matrix(
            [[self.phi_R], [self.phi_L]]) + Matrix([[1], [0]]) * self.theta_0)[0, 0]

        self.beta = (1/2 * Matrix([[1, -1], [1, 1]]) * Matrix(
            [[self.phi_R], [self.phi_L]]) + Matrix([[1], [0]]) * self.theta_0)[1, 0]

        # vector in reference frame
        self.OG_ref = smp.symbols('OG_ref', cls=smp.Function)
        self.OAR_ref = smp.symbols('OAR_ref', cls=smp.Function)
        self.OBR_ref = smp.symbols('OBR_ref', cls=smp.Function)
        self.OCR_ref = smp.symbols('OCR_ref', cls=smp.Function)
        self.ODR_ref = smp.symbols('ODR_ref', cls=smp.Function)
        self.OER_ref = smp.symbols('OER_ref', cls=smp.Function)
        self.OFR_ref = smp.symbols('OFR_ref', cls=smp.Function)

        self.OAL_ref = smp.symbols('OAL_ref', cls=smp.Function)
        self.OBL_ref = smp.symbols('OBL_ref', cls=smp.Function)
        self.OCL_ref = smp.symbols('OCL_ref', cls=smp.Function)
        self.ODL_ref = smp.symbols('ODL_ref', cls=smp.Function)
        self.OEL_ref = smp.symbols('OEL_ref', cls=smp.Function)
        self.OFL_ref = smp.symbols('OFL_ref', cls=smp.Function)

        self.OG_ref = self.OG_ref(self.t)
        self.OAR_ref = self.OAR_ref(self.t)
        self.OBR_ref = self.OBR_ref(self.t)
        self.OCR_ref = self.OCR_ref(self.t)
        self.ODR_ref = self.ODR_ref(self.t)
        self.OER_ref = self.OER_ref(self.t)
        self.OFR_ref = self.OFR_ref(self.t)

        self.OAL_ref = self.OAL_ref(self.t)
        self.OBL_ref = self.OBL_ref(self.t)
        self.OCL_ref = self.OCL_ref(self.t)
        self.ODL_ref = self.ODL_ref(self.t)
        self.OEL_ref = self.OEL_ref(self.t)
        self.OFL_ref = self.OFL_ref(self.t)

        # vector in theta-beta frame
        self.O = Matrix([[0], [0]])
        self.OG = smp.symbols('OG', cls=smp.Function)
        self.OAR = smp.symbols('OAR', cls=smp.Function)
        self.OBR = smp.symbols('OBR', cls=smp.Function)
        self.OCR = smp.symbols('OCR', cls=smp.Function)
        self.ODR = smp.symbols('ODR', cls=smp.Function)
        self.OER = smp.symbols('OER', cls=smp.Function)
        self.OFR = smp.symbols('OFR', cls=smp.Function)

        self.OAL = smp.symbols('OAL', cls=smp.Function)
        self.OBL = smp.symbols('OBL', cls=smp.Function)
        self.OCL = smp.symbols('OCL', cls=smp.Function)
        self.ODL = smp.symbols('ODL', cls=smp.Function)
        self.OEL = smp.symbols('OEL', cls=smp.Function)
        self.OFL = smp.symbols('OFL', cls=smp.Function)

        self.OG = self.OG(self.t)
        self.OAR = self.OAR(self.t)
        self.OBR = self.OBR(self.t)
        self.OCR = self.OCR(self.t)
        self.ODR = self.ODR(self.t)
        self.OER = self.OER(self.t)
        self.OFR = self.OFR(self.t)

        self.OAL = self.OAL(self.t)
        self.OBL = self.OBL(self.t)
        self.OCL = self.OCL(self.t)
        self.ODL = self.ODL(self.t)
        self.OEL = self.OEL(self.t)
        self.OFL = self.OFL(self.t)

        self.JacobianOG = 0
        pass

    def foward_kinematic(self):
        self.OAR_ref = Matrix([[self.l1_*smp.sin(self.theta)],
                               [self.l1_*smp.cos(self.theta)]])

        self.OBR_ref = Matrix([[self.R*smp.sin(self.theta)],
                               [self.R*smp.cos(self.theta)]])

        alpha_1 = smp.pi - self.theta
        alpha_2 = smp.asin((self.l1_/(self.l5_+self.l6_)) * smp.sin(alpha_1))

        self.OER_ref = Matrix(
            [[0], [self.l1_*smp.cos(self.theta)-(self.l5_+self.l6_)*smp.cos(alpha_2)]])

        self.ODR_ref = self.l5_/(self.l5_+self.l6_) * \
            self.OER_ref + self.l6_/(self.l5_+self.l6_) * self.OAR_ref

        # Derive vector OC
        BD = self.ODR_ref - self.OBR_ref
        DB = -BD
        # l_BD
        l_BD = smp.sqrt(Transpose(BD) * BD)[0, 0]
        # alpha_3 defined as angle BDC
        cos_alpha_3 = (l_BD**2 + self.l4_**2 - self.l3_**2) / \
            (2 * l_BD * self.l4_)

        sin_alpha_3 = smp.sqrt(1 - cos_alpha_3**2)

        rot_alpha_3 = Matrix(
            [[cos_alpha_3, sin_alpha_3], [-sin_alpha_3, cos_alpha_3]])

        DC = (self.l4_/l_BD) * rot_alpha_3 * DB

        self.OCR_ref = self.ODR_ref + DC

        # Derive vector OF
        CB = self.OBR_ref - self.OCR_ref
        # alpha_4 defined as angle BCF
        alpha_4 = smp.acos(self.l3_/(2*self.R)) + smp.acos(self.l7_/(2*self.R))
        # rotate CB to CF direction
        rot_alpha_4 = Matrix([[smp.cos(alpha_4), -smp.sin(alpha_4)],
                              [smp.sin(alpha_4), smp.cos(alpha_4)]])

        self.l_BF = smp.sqrt(self.l3_**2 + self.l7_**2 - 2 *
                             self.l3_*self.l7_*smp.cos(alpha_4))
        l_CB = smp.sqrt(Transpose(CB) * CB)[0, 0]

        CF = (self.l7_/l_CB) * rot_alpha_4 * CB
        self.OFR_ref = self.OCR_ref + CF

        # Derive vector OG
        # alpha_5 defined as angle OGF
        # alpha_6 defined as angle GOF
        l_OF = smp.sqrt(Transpose(self.OFR_ref) * self.OFR_ref)[0, 0]
        sin_alpha_6 = self.OFR_ref[0, 0]/l_OF
        sin_alpha_5 = (l_OF/self.l8_) * sin_alpha_6
        cos_alpha_5 = smp.sqrt(1 - sin_alpha_5**2)
        self.OG_ref = Matrix(
            [[0], [self.OFR_ref[1, 0] - self.l8_ * cos_alpha_5]])

        # Derive Left Hand Plane Vectors
        mirror_mat = Matrix([[-1, 0],
                             [0, 1]])

        self.OAL_ref = mirror_mat * self.OAR_ref
        self.OBL_ref = mirror_mat * self.OBR_ref
        self.OCL_ref = mirror_mat * self.OCR_ref
        self.ODL_ref = mirror_mat * self.ODR_ref
        self.OEL_ref = mirror_mat * self.OER_ref
        self.OFL_ref = mirror_mat * self.OFR_ref

        # transform to beta-theta frame
        rot_beta = Matrix([[smp.cos(self.beta), smp.sin(self.beta)],
                          [-smp.sin(self.beta), smp.cos(self.beta)]])

        self.OAR = rot_beta * self.OAR_ref
        self.OBR = rot_beta * self.OBR_ref
        self.OCR = rot_beta * self.OCR_ref
        self.ODR = rot_beta * self.ODR_ref
        self.OER = rot_beta * self.OER_ref
        self.OFR = rot_beta * self.OFR_ref
        self.OAL = rot_beta * self.OAL_ref
        self.OBL = rot_beta * self.OBL_ref
        self.OCL = rot_beta * self.OCL_ref
        self.ODL = rot_beta * self.ODL_ref
        self.OEL = rot_beta * self.OEL_ref
        self.OFL = rot_beta * self.OFL_ref
        self.OG = rot_beta * self.OG_ref

        self.JacobianOG = self.OG.jacobian(self.vec_phi)
        J_OG = smp.lambdify(self.vec_phi, self.JacobianOG)
        with open('./serialized_object/J_OG.pkl', 'wb') as d:
            dill.dump(J_OG, d)


if __name__ == '__main__':
    k = ls.LegSymbol()
    leg = LegModel(k)
    leg.foward_kinematic()
