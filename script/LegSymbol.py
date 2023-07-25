import math
import numpy as np
import sympy as smp
from sympy import symbols, Matrix, Transpose


class LegSymbol:
    def __init__(self) -> None:
        self.t = smp.symbols('t')
        # phi_R, phi_L terms
        self.phi_R = smp.symbols('phi_R', cls=smp.Function)
        self.phi_L = smp.symbols('phi_L', cls=smp.Function)
        self.phi_R = self.phi_R(self.t)
        self.phi_L = self.phi_L(self.t)
        self.dphi_R = smp.diff(self.phi_R, self.t)
        self.dphi_L = smp.diff(self.phi_L, self.t)
        self.ddphi_R = smp.diff(self.dphi_R, self.t)
        self.ddphi_L = smp.diff(self.dphi_L, self.t)

        # theta, beta terms
        self.theta = smp.symbols('theta', cls=smp.Function)
        self.beta = smp.symbols('beta', cls=smp.Function)
        self.theta = self.theta(self.t)
        self.beta = self.beta(self.t)
        pass
