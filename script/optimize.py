from cpg_fsm import *
from scipy.optimize import differential_evolution
from scipy.optimize import LinearConstraint, Bounds
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
from scipy.optimize import shgo
import datetime
import os
import pygad


class corgiOptimize:
    def __init__(self) -> None:
        self.freq = 100
        self.L = 0.2
        self.H_st = 0.2  # stance height
        self.T_sw = 0.4  # swing time
        self.px_init = self.L / 4
        self.mx_init = -self.L / 4

        # Bezier Curve Constant
        self.bz_pointnum = 20
        self.cons_ws_shrink_r = 0.01

        # Optimization Setup
        self.liftstate = []
        self.obj_itercnt = 0

        # Optimization boundary
        self.H_b = (0.05, 0.06)
        self.dH_b = (0.00, 0.02)
        self.L1_b = (0.05, 1)
        self.L2_b = (-0.1, 0.1)
        self.L3_b = (0.05, 1)
        self.L4_b = (-0.1, 0.1)

        # Optimization initial guess
        self.bp_init_guess = np.array(
            [
                [0.05, 0.01, 0.1, 0.01, 0.1, 0.01],
                [0.05, 0.01, 0.1, 0.01, 0.1, 0.01],
                [0.05, 0.01, 0.1, 0.01, 0.1, 0.01],
                [0.05, 0.01, 0.1, 0.01, 0.1, 0.01],
            ]
        )

        # Optimization Constraint
        self.innerCons = {"type": "ineq", "fun": self.innerWsConstraint}
        self.outerCons = {"type": "ineq", "fun": self.outerWsConstraint}
        self.centerlineCons = {"type": "ineq", "fun": self.forehindWsConstraint}
        self.bezCons = {"type": "ineq", "fun": self.bezierProfileConstraint}
        self.velCons = {"type": "eq", "fun": self.velocityConstraint}
        self.Cons = [self.innerCons, self.outerCons, self.centerlineCons, self.bezCons, self.velCons]

        # Optimization Cost
        self.C_s = 3
        self.C_u = 0.5
        self.opt_result = None

        self.generate_cycle = 4 * 6
        self.total_t = 30
        self.sbrio_freq = 400

    def getLiftoffState(self):
        tb_1 = lk.InverseKinematicsPoly(np.array([[self.px_init], [-self.H_st]]))
        tb_2 = lk.InverseKinematicsPoly(np.array([[self.mx_init], [-self.H_st]]))
        init_A_tb = tb_1
        init_B_tb = tb_1
        init_C_tb = tb_2
        init_D_tb = tb_2

        FSM = FiniteStateMachine(self.freq)
        CORGI = Corgi(self.freq, FSM)
        CORGI.step_length = self.L
        CORGI.stance_height = self.H_st
        CORGI.swing_time = self.T_sw
        CORGI.weight_s = self.C_s
        CORGI.weight_u = self.C_u
        CORGI.total_cycle = 4
        CORGI.total_time = self.total_t
        CORGI.setInitPhase(init_A_tb, init_B_tb, init_C_tb, init_D_tb)
        CORGI.move()
        A_liftstate = CORGI.lift_state[0]
        B_liftstate = CORGI.lift_state[2]
        C_liftstate = CORGI.lift_state[1]
        D_liftstate = CORGI.lift_state[3]
        self.liftstate = [A_liftstate, B_liftstate, C_liftstate, D_liftstate]

    def getBezierControlPoint(self, c0, bp):
        h, dh, dL1, dL2, dL3, dL4 = bp
        L = self.L
        c0 = np.array([0, 0])
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
        return c_set

    def getBezierCurve(self, bezier_profile):
        c_set = self.getBezierControlPoint(np.array([0, 0]), bezier_profile)
        t_points = np.linspace(0, 1, self.bz_pointnum)
        curve = Bezier.Curve(t_points, c_set)
        return curve

    def innerWsConstraint(self, bezier_profiles):
        step_len = self.L
        shrink_margin = self.cons_ws_shrink_r
        point_num = self.bz_pointnum
        liftstates = self.liftstate

        bps = bezier_profiles.reshape(4, -1)
        t_ = np.linspace(0, 1, point_num).reshape(point_num, -1)
        subs = []

        for i in range(4):
            b_curve = self.getBezierCurve(bps[i])
            v_OG = liftstates[i][2].copy()
            if i == 0 or i == 3:
                v_OG[0, 0] *= -1
            hip1 = -1 * v_OG
            hip1s = np.repeat(hip1.T, repeats=point_num, axis=0)
            hip_t = hip1s + t_ @ np.array([[self.L / 4, 0]])

            dv = hip_t - b_curve
            d = np.linalg.norm(dv, axis=1).reshape(point_num, -1)

            # inner circle constraint d - (0.1 + shrink_margin) >0
            r_ = (0.1 + shrink_margin) * np.ones([point_num, 1])
            subs.append(d - r_)

        cons = np.vstack((subs[0], subs[1]))
        cons = np.vstack((cons, subs[2]))
        cons = np.vstack((cons, subs[3]))
        cons = cons.reshape(1, -1)[0]
        # cons >= 0 to fit the constraint
        return cons

    def outerWsConstraint(self, bezier_profiles):
        step_len = self.L
        shrink_margin = self.cons_ws_shrink_r
        point_num = self.bz_pointnum
        liftstates = self.liftstate

        bps = bezier_profiles.reshape(4, -1)
        t_ = np.linspace(0, 1, point_num).reshape(point_num, -1)
        subs = []

        for i in range(4):
            b_curve = self.getBezierCurve(bps[i])
            v_OG = liftstates[i][2].copy()
            if i == 0 or i == 3:
                v_OG[0, 0] *= -1
            hip1 = -1 * v_OG
            hip1s = np.repeat(hip1.T, repeats=point_num, axis=0)
            hip_t = hip1s + t_ @ np.array([[self.L / 4, 0]])

            dv = hip_t - b_curve
            d = np.linalg.norm(dv, axis=1).reshape(point_num, -1)

            # outer circle constraint (0.3428 - shrink_margin) - d >0
            r_ = (0.3428 - shrink_margin) * np.ones([point_num, 1])
            subs.append(r_ - d)

        cons = np.vstack((subs[0], subs[1]))
        cons = np.vstack((cons, subs[2]))
        cons = np.vstack((cons, subs[3]))
        cons = cons.reshape(1, -1)[0]
        # cons >= 0 to fit the constraint
        return cons

    def forehindWsConstraint(self, bezier_profiles):
        shrink_margin = self.cons_ws_shrink_r
        point_num = self.bz_pointnum
        liftstates = self.liftstate

        bps = bezier_profiles.reshape(4, -1)
        t_ = np.linspace(0, 1, point_num).reshape(point_num, -1)
        subs = []

        for i in range(4):
            b_curve = self.getBezierCurve(bps[i])
            v_OG = liftstates[i][2].copy()
            if i == 0 or i == 3:
                v_OG[0, 0] *= -1
            hip1 = -1 * v_OG
            hip1s = np.repeat(hip1.T, repeats=point_num, axis=0)
            hip_t = hip1s + t_ @ np.array([[self.L / 4, 0]])
            dv = hip_t - b_curve
            d = np.linalg.norm(dv, axis=1).reshape(point_num, -1)
            if i == 0 or i == 1:
                # PointG_x >= center_line_x + margin --> PointG_x - (center_line_x + margin) >= 0
                center_line_x = (
                    hip_t[:, 0].reshape(point_num, -1)
                    - 0.222 * np.ones([point_num, 1])
                    + shrink_margin * np.ones([point_num, 1])
                )
                b_curve_x = b_curve[:, 0].reshape(point_num, -1)
                subs.append(b_curve_x - center_line_x)
            else:
                center_line_x = (
                    hip_t[:, 0].reshape(point_num, -1)
                    + 0.222 * np.ones([point_num, 1])
                    - shrink_margin * np.ones([point_num, 1])
                )
                b_curve_x = b_curve[:, 0].reshape(point_num, -1)
                subs.append(center_line_x - b_curve_x)
        cons = np.vstack((subs[0], subs[1]))
        cons = np.vstack((cons, subs[2]))
        cons = np.vstack((cons, subs[3]))
        cons = cons.reshape(1, -1)[0]
        # cons >= 0 to fit the constraint
        return cons

    def innerWsPFConstraint(self, bezier_profiles):
        shrink_margin = self.cons_ws_shrink_r
        point_num = self.bz_pointnum
        liftstates = self.liftstate

        bps = bezier_profiles.reshape(4, -1)
        t_ = np.linspace(0, 1, point_num).reshape(point_num, -1)
        subs = []
        pot_list = []

        Q_ = 0.2 + shrink_margin
        W_ = 1

        for i in range(4):
            b_curve = self.getBezierCurve(bps[i])
            v_OG = liftstates[i][2].copy()
            if i == 0 or i == 3:
                v_OG[0, 0] *= -1
            hip1 = -1 * v_OG
            hip1s = np.repeat(hip1.T, repeats=point_num, axis=0)
            hip_t = hip1s + t_ @ np.array([[self.L / 4, 0]])
            dv = hip_t - b_curve
            d = np.linalg.norm(dv, axis=1).reshape(point_num, -1)

            pot_sum = 0
            for d_ in d:
                if d_ >= Q_:
                    U = 0
                else:
                    if d_ < 0.0001:
                        d__ = 0.0001
                    else:
                        d__ = d_
                    U = 0.5 * W_ * (1 / d__ - 1 / Q_) ** 2
                pot_sum += U
            pot_list.append(pot_sum[0])
        return np.array(pot_list)

    def outerWsPFConstraint(self, bezier_profiles):
        shrink_margin = self.cons_ws_shrink_r
        point_num = self.bz_pointnum
        liftstates = self.liftstate

        bps = bezier_profiles.reshape(4, -1)
        t_ = np.linspace(0, 1, point_num).reshape(point_num, -1)
        subs = []
        pot_list = []

        W_ = 100
        R_bnd = 0.3428 - shrink_margin

        for i in range(4):
            b_curve = self.getBezierCurve(bps[i])
            v_OG = liftstates[i][2].copy()
            if i == 0 or i == 3:
                v_OG[0, 0] *= -1
            hip1 = -1 * v_OG
            hip1s = np.repeat(hip1.T, repeats=point_num, axis=0)
            hip_t = hip1s + t_ @ np.array([[self.L / 4, 0]])

            dv = hip_t - b_curve
            d = np.linalg.norm(dv, axis=1).reshape(point_num, -1)

            pot_sum = 0
            for d_ in d:
                D_ = abs(d_[0] - R_bnd - shrink_margin)
                if D_ < 0.001:
                    D_ = 0.001
                if d_ < R_bnd:
                    U = 0
                else:
                    U = 0.5 * W_ * (D_) ** 2
                pot_sum += U
            pot_list.append(pot_sum)
        return np.array(pot_list)

    def forehindWsPFConstraint(self, bezier_profiles):
        shrink_margin = self.cons_ws_shrink_r
        point_num = self.bz_pointnum
        liftstates = self.liftstate

        bps = bezier_profiles.reshape(4, -1)
        t_ = np.linspace(0, 1, point_num).reshape(point_num, -1)
        subs = []

        for i in range(4):
            b_curve = self.getBezierCurve(bps[i])
            v_OG = liftstates[i][2].copy()
            if i == 0 or i == 3:
                v_OG[0, 0] *= -1
            hip1 = -1 * v_OG
            hip1s = np.repeat(hip1.T, repeats=point_num, axis=0)
            hip_t = hip1s + t_ @ np.array([[self.L / 4, 0]])
            dv = hip_t - b_curve
            d = np.linalg.norm(dv, axis=1).reshape(point_num, -1)
            if i == 0 or i == 1:
                # PointG_x >= center_line_x + margin --> PointG_x - (center_line_x + margin) >= 0
                center_line_x = (
                    hip_t[:, 0].reshape(point_num, -1)
                    - 0.222 * np.ones([point_num, 1])
                    + shrink_margin * np.ones([point_num, 1])
                )
                b_curve_x = b_curve[:, 0].reshape(point_num, -1)
                subs.append(b_curve_x - center_line_x)
            else:
                center_line_x = (
                    hip_t[:, 0].reshape(point_num, -1)
                    + 0.222 * np.ones([point_num, 1])
                    - shrink_margin * np.ones([point_num, 1])
                )
                b_curve_x = b_curve[:, 0].reshape(point_num, -1)
                subs.append(center_line_x - b_curve_x)
        cons = np.vstack((subs[0], subs[1]))
        cons = np.vstack((cons, subs[2]))
        cons = np.vstack((cons, subs[3]))
        cons = cons.reshape(1, -1)[0]
        # cons >= 0 to fit the constraint
        return cons

    def bezierProfileConstraint(self, bezier_profiles):
        bps = bezier_profiles.reshape(4, -1)
        cons = []
        for i in range(4):
            cset = self.getBezierControlPoint(np.array([0, 0]), bps[i])
            c2_x = cset[2, 0]
            c5_x = cset[5, 0]
            c8_x = cset[8, 0]
            # c2_x < c5_x, c8_x > c5_x
            cons.append(c5_x - c2_x)
            cons.append(c8_x - c5_x)
        return np.array(cons)

    def velocityConstraint(self, bezier_profiles):
        bps = bezier_profiles.reshape(4, -1)
        cons = []
        dt = self.T_sw / self.bz_pointnum
        v_stance = np.array([-(self.L / 4) / self.T_sw, 0])
        cons = []
        for i in range(4):
            b_curve = self.getBezierCurve(bps[i])
            v_liftoff = (b_curve[1] - b_curve[0]) / dt
            v_landing = (b_curve[-1] - b_curve[-2]) / dt

            dv_liftoff = v_liftoff - v_stance
            dv_landing = v_landing - v_stance
            cons.append(dv_liftoff)
            cons.append(dv_landing)
        return np.array(cons).reshape(1, -1)[0]

    def allConstraint(self, bezier_profile):
        c1 = self.innerWsConstraint(bezier_profile)
        c2 = self.outerWsConstraint(bezier_profile)
        c3 = self.forehindWsConstraint(bezier_profile)
        c4 = self.bezierProfileConstraint(bezier_profile)
        c5 = self.velocityConstraint(bezier_profile)
        c = np.hstack((c1, c2))
        # c = np.hstack((c, c3))
        # c = np.hstack((c, c4))
        c = np.hstack((c, c5))
        return c

    def objective(self, bezier_profiles):
        bp = bezier_profiles.reshape(4, -1)
        print(bp[0])
        print(bp[1])
        print(bp[2])
        print(bp[3])

        tb_1 = lk.InverseKinematicsPoly(np.array([[self.px_init], [-self.H_st]]))
        tb_2 = lk.InverseKinematicsPoly(np.array([[self.mx_init], [-self.H_st]]))
        init_A_tb = tb_1
        init_B_tb = tb_1
        init_C_tb = tb_2
        init_D_tb = tb_2

        FSM = FiniteStateMachine(self.freq)
        CORGI = Corgi(self.freq, FSM)
        CORGI.step_length = self.L
        CORGI.stance_height = self.H_st
        CORGI.swing_time = self.T_sw
        CORGI.weight_s = self.C_s
        CORGI.weight_u = self.C_u
        CORGI.total_cycle = 4
        CORGI.total_time = self.total_t
        CORGI.setInitPhase(init_A_tb, init_B_tb, init_C_tb, init_D_tb)
        CORGI.move(swing_profile=bp)
        self.obj_itercnt += 1

        print("iter", self.obj_itercnt, "cost", CORGI.cost)
        return CORGI.cost

    def multiObjective(self, bezier_profiles):
        bp = bezier_profiles.reshape(4, -1)
        tb_1 = lk.InverseKinematicsPoly(np.array([[self.px_init], [-self.H_st]]))
        tb_2 = lk.InverseKinematicsPoly(np.array([[self.mx_init], [-self.H_st]]))
        init_A_tb = tb_1
        init_B_tb = tb_1
        init_C_tb = tb_2
        init_D_tb = tb_2

        FSM = FiniteStateMachine(self.freq)
        CORGI = Corgi(self.freq, FSM)
        CORGI.step_length = self.L
        CORGI.stance_height = self.H_st
        CORGI.swing_time = self.T_sw
        CORGI.weight_s = self.C_s
        CORGI.weight_u = self.C_u
        CORGI.total_cycle = 4
        CORGI.total_time = self.total_t
        CORGI.setInitPhase(init_A_tb, init_B_tb, init_C_tb, init_D_tb)
        CORGI.move(swing_profile=bp)
        self.obj_itercnt += 1

        print("iter", self.obj_itercnt, "cost", CORGI.cost)
        return [CORGI.cost_s, CORGI.cost_u]

    def run_minimize(self):
        # fmt:off
        bnds = (self.H_b, self.dH_b, self.L1_b, self.L2_b, self.L3_b, self.L4_b,
                self.H_b, self.dH_b, self.L1_b, self.L2_b, self.L3_b, self.L4_b,
                self.H_b, self.dH_b, self.L1_b, self.L2_b, self.L3_b, self.L4_b,
                self.H_b, self.dH_b, self.L1_b, self.L2_b, self.L3_b, self.L4_b)
        # fmt:on
        self.opt_result = minimize(
            self.objective,
            self.bp_init_guess.reshape(1, -1),
            method="SLSQP",
            bounds=bnds,
            constraints=self.Cons,
            options={"disp": True, "maxiter": 100},
        )
        self.plotResult(self.opt_result)

    def run_evolution(self):
        # fmt:off
        bnds = (self.H_b, self.dH_b, self.L1_b, self.L2_b, self.L3_b, self.L4_b,
                self.H_b, self.dH_b, self.L1_b, self.L2_b, self.L3_b, self.L4_b,
                self.H_b, self.dH_b, self.L1_b, self.L2_b, self.L3_b, self.L4_b,
                self.H_b, self.dH_b, self.L1_b, self.L2_b, self.L3_b, self.L4_b)
        # fmt:on
        nlc1_shape = self.innerWsPFConstraint(self.bp_init_guess).shape
        nlc4_shape = self.bezierProfileConstraint(self.bp_init_guess).shape
        nlc5_shape = self.velocityConstraint(self.bp_init_guess).shape
        print(nlc1_shape)
        print(nlc5_shape)

        # nlc1 = NonlinearConstraint(self.innerWsConstraint, np.zeros(nlc1_shape), np.inf * np.ones(nlc1_shape))
        # nlc2 = NonlinearConstraint(self.outerWsConstraint, np.zeros(nlc1_shape), np.inf * np.ones(nlc1_shape))
        nlc1 = NonlinearConstraint(self.innerWsPFConstraint, np.zeros([1, 4]), np.zeros([1, 4]))
        nlc2 = NonlinearConstraint(self.outerWsPFConstraint, np.zeros([1, 4]), np.zeros([1, 4]))
        # nlc3 = NonlinearConstraint(self.forehindWsConstraint, np.zeros(nlc1_shape), np.inf * np.ones(nlc1_shape))
        # nlc4 = NonlinearConstraint(self.bezierProfileConstraint, np.zeros(nlc4_shape), np.inf * np.ones(nlc4_shape))
        nlc5 = NonlinearConstraint(self.velocityConstraint, np.zeros(nlc5_shape), np.inf * np.ones(nlc5_shape))
        # nlc = NonlinearConstraint(self.allConstraint, np.zeros(40 * 2), np.inf * np.ones(40 * 2))

        nlcs = [nlc1, nlc2, nlc5]
        # nlcs = [nlc1, nlc2, nlc3, nlc5]

        # self.opt_result = differential_evolution(
        #     self.objective,
        #     x0=self.bp_init_guess.reshape(1, -1),
        #     bounds=bnds,
        #     popsize=1,
        #     recombination=0.5,
        #     seed=1,
        #     workers=-1,
        #     disp=True,
        #     init="random",
        #     constraints=nlc1,
        # )

        # self.opt_result = shgo(
        #     self.objective,
        #     bounds=bnds,
        #     minimizer_kwargs={"method": "SLSQP"},
        #     constraints=nlcs,
        # )
        self.plotResult(self.opt_result)

    def fitness_function(self, ga_instance, bezier_profiles, solution_idx):
        bp = bezier_profiles.reshape(4, -1)
        # print(bp)
        tb_1 = lk.InverseKinematicsPoly(np.array([[self.px_init], [-self.H_st]]))
        tb_2 = lk.InverseKinematicsPoly(np.array([[self.mx_init], [-self.H_st]]))
        init_A_tb = tb_1
        init_B_tb = tb_1
        init_C_tb = tb_2
        init_D_tb = tb_2

        FSM = FiniteStateMachine(self.freq)
        CORGI = Corgi(self.freq, FSM)
        CORGI.step_length = self.L
        CORGI.stance_height = self.H_st
        CORGI.swing_time = self.T_sw
        CORGI.weight_s = self.C_s
        CORGI.weight_u = self.C_u
        CORGI.total_cycle = 4
        CORGI.total_time = self.total_t
        CORGI.setInitPhase(init_A_tb, init_B_tb, init_C_tb, init_D_tb)
        CORGI.move(swing_profile=bp)

        # violation penalty
        penalty = 0
        violate_cnt = 0
        violate_sum = 0
        cons = self.allConstraint(bezier_profiles)
        for i in cons:
            if i < 0:
                violate_sum += i
                # violate_cnt += 1

        # if (cons >= -0.5 * np.ones(cons.shape)).all():
        #     penalty = 0
        # else:
        #     violate_cnt += 1
        #     penalty = -100
        #     print("iter", self.obj_itercnt, "cost", penalty)
        #     # print(cons)
        #     return -100

        fitness = -3 * CORGI.cost + violate_sum

        self.obj_itercnt += 1
        print("iter", self.obj_itercnt, "cost", fitness, "violate_sum", violate_sum)
        return fitness

    def run_GA(self):
        H_b = [0.04, 0.07]
        dH_b = [0.00, 0.02]
        L1_b = [0.01, 1]
        L2_b = [-0.5, 0.5]
        L3_b = [0.01, 1]
        L4_b = [-0.5, 0.5]
        # fmt:off
        gene_space = [H_b, dH_b, L1_b, L2_b, L3_b, L4_b,
                      H_b, dH_b, L1_b, L2_b, L3_b, L4_b,
                      H_b, dH_b, L1_b, L2_b, L3_b, L4_b,
                      H_b, dH_b, L1_b, L2_b, L3_b, L4_b]
        initial_pop = []
        bp_init = [0.05, 0.01, 0.05, 0.01, 0.03, 0.01,
                   0.05, 0.01, 0.05, 0.01, 0.03, 0.01,
                   0.05, 0.01, 0.05, 0.01, 0.03, 0.01,
                   0.05, 0.01, 0.05, 0.01, 0.03, 0.01]
        # fmt:on
        for i in range(32):
            initial_pop.append(list(bp_init))

        ga_instance = pygad.GA(
            initial_population=initial_pop,
            num_generations=400,
            num_parents_mating=16,
            fitness_func=self.fitness_function,
            sol_per_pop=64,
            num_genes=24,
            gene_space=gene_space,
            parent_selection_type="rws",
            keep_parents=-1,
            crossover_type="uniform",
            crossover_probability=0.5,
            mutation_probability=0.4,
            mutation_type="random",
            mutation_percent_genes="default",
            parallel_processing=["process", 12],
        )
        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    def plotResult(self, result):
        opt_bez = result
        opt_bez = opt_bez.reshape(4, -1)
        print("---")
        print("Optimized Bezier Profile:")
        print(opt_bez)

        fig, ax = plt.subplots(2, 2)
        A_cset = self.getBezierControlPoint(np.array([0, 0]), opt_bez[0])
        t_points = np.linspace(0, 1, 100)
        A_curve = Bezier.Curve(t_points, A_cset)

        B_cset = self.getBezierControlPoint(np.array([0, 0]), opt_bez[1])
        B_curve = Bezier.Curve(t_points, B_cset)

        C_cset = self.getBezierControlPoint(np.array([0, 0]), opt_bez[2])
        C_curve = Bezier.Curve(t_points, C_cset)

        D_cset = self.getBezierControlPoint(np.array([0, 0]), opt_bez[3])
        D_curve = Bezier.Curve(t_points, D_cset)

        ax[0][0].plot(A_curve[:, 0], A_curve[:, 1])
        ax[0][0].plot(A_cset[:, 0], A_cset[:, 1], "-o")

        ax[0][1].plot(B_curve[:, 0], B_curve[:, 1])
        ax[0][1].plot(B_cset[:, 0], B_cset[:, 1], "-o")

        ax[1][0].plot(C_curve[:, 0], C_curve[:, 1])
        ax[1][0].plot(C_cset[:, 0], C_cset[:, 1], "-o")

        ax[1][1].plot(D_curve[:, 0], D_curve[:, 1])
        ax[1][1].plot(D_cset[:, 0], D_cset[:, 1], "-o")
        plt.show()

    def saveData(self, filepath="./csv_trajectory/20230828/", idx=1):
        filename_check = False
        file_idx = idx
        sbrio_filename = str(datetime.date.today()) + "_traj_400_" + str(file_idx) + ".csv"
        param_filename = str(datetime.date.today()) + "_param_400_" + str(file_idx) + ".csv"
        while not filename_check:
            if os.path.isfile(filepath + sbrio_filename) or os.path.isfile(filepath + param_filename):
                file_idx += 1
                sbrio_filename = str(datetime.date.today()) + "_traj_400_" + str(file_idx) + ".csv"
                param_filename = str(datetime.date.today()) + "_param_400_" + str(file_idx) + ".csv"
            else:
                csv_filepath = filepath + sbrio_filename
                param_filepath = filepath + param_filename
                filename_check = True

        webot_filepath = "/home/guanlunlu/corgi_webots/controllers/supervisor/" + sbrio_filename

        with open(param_filepath, "w") as f:
            f.write("loop_freq," + str(self.freq) + "\n")
            f.write("Step_length," + str(self.L) + "\n")
            f.write("swing_time," + str(self.T_sw) + "\n")
            f.write("C_s," + str(self.C_s) + "\n")
            f.write("C_u," + str(self.C_u) + "\n")
            f.write("px_init," + str(self.px_init) + "\n")
            f.write("mx_init," + str(self.mx_init) + "\n")
            f.write("cycle," + str(self.generate_cycle) + "\n")
            f.write("total_t," + str(self.total_t) + "\n")
            f.write("init_bezier_profile, " + repr(self.bp_init_guess) + "\n")
            f.write("optimize_bezier_profile, " + repr(self.opt_result) + "\n")
            f.write("\n")
            f.write(repr(self.opt_result))
            f.write("\n")

        opt_bez = self.opt_result.x.reshape(4, -1)
        # Export CSV for sbrio
        tb_1 = lk.InverseKinematicsPoly(np.array([[self.px_init], [-self.H_st]]))
        tb_2 = lk.InverseKinematicsPoly(np.array([[self.mx_init], [-self.H_st]]))
        init_A_tb = tb_1
        init_B_tb = tb_1
        init_C_tb = tb_2
        init_D_tb = tb_2
        FSM = FiniteStateMachine(self.sbrio_freq)
        CORGI = Corgi(self.sbrio_freq, FSM)
        CORGI.step_length = self.L
        CORGI.stance_height = self.H_st
        CORGI.swing_time = self.T_sw
        CORGI.weight_s = self.C_s
        CORGI.weight_u = self.C_u
        CORGI.total_cycle = self.generate_cycle
        CORGI.total_time = self.total_t
        CORGI.setInitPhase(init_A_tb, init_B_tb, init_C_tb, init_D_tb)
        CORGI.move(swing_profile=opt_bez)
        CORGI.exportCSV(csv_filepath)
        print("sbrio exported to", csv_filepath)

        # Export CSV for webot
        tb_1 = lk.InverseKinematicsPoly(np.array([[self.px_init], [-self.H_st]]))
        tb_2 = lk.InverseKinematicsPoly(np.array([[self.mx_init], [-self.H_st]]))
        init_A_tb = tb_1
        init_B_tb = tb_1
        init_C_tb = tb_2
        init_D_tb = tb_2
        FSM = FiniteStateMachine(40)
        CORGI = Corgi(40, FSM)
        CORGI.step_length = self.L
        CORGI.stance_height = self.H_st
        CORGI.swing_time = self.T_sw
        CORGI.weight_s = self.C_s
        CORGI.weight_u = self.C_u
        CORGI.total_cycle = self.generate_cycle
        CORGI.total_time = self.total_t
        CORGI.setInitPhase(init_A_tb, init_B_tb, init_C_tb, init_D_tb)
        CORGI.move(swing_profile=opt_bez)
        CORGI.exportCSV(webot_filepath)
        print("webot exported", webot_filepath)

    def exportBezierData(self, bp, filepath="./csv_trajectory/20230828/", idx=1):
        filename_check = False
        file_idx = idx
        sbrio_filename = str(datetime.date.today()) + "_traj_400_" + str(file_idx) + ".csv"
        param_filename = str(datetime.date.today()) + "_param_400_" + str(file_idx) + ".csv"
        while not filename_check:
            if os.path.isfile(filepath + sbrio_filename) or os.path.isfile(filepath + param_filename):
                file_idx += 1
                sbrio_filename = str(datetime.date.today()) + "_traj_400_" + str(file_idx) + ".csv"
                param_filename = str(datetime.date.today()) + "_param_400_" + str(file_idx) + ".csv"
            else:
                csv_filepath = filepath + sbrio_filename
                param_filepath = filepath + param_filename
                filename_check = True

        webot_filepath = "/home/guanlunlu/corgi_webots/controllers/supervisor/" + sbrio_filename

        with open(param_filepath, "w") as f:
            f.write("loop_freq," + str(self.freq) + "\n")
            f.write("Step_length," + str(self.L) + "\n")
            f.write("swing_time," + str(self.T_sw) + "\n")
            f.write("C_s," + str(self.C_s) + "\n")
            f.write("C_u," + str(self.C_u) + "\n")
            f.write("px_init," + str(self.px_init) + "\n")
            f.write("mx_init," + str(self.mx_init) + "\n")
            f.write("cycle," + str(self.generate_cycle) + "\n")
            f.write("total_t," + str(self.total_t) + "\n")
            f.write("init_bezier_profile, " + repr(self.bp_init_guess) + "\n")
            f.write("optimize_bezier_profile, " + repr(bp) + "\n")
            f.write("\n")
            f.write(repr(self.opt_result))
            f.write("\n")

        opt_bez = bp.reshape(4, -1)
        # Export CSV for sbrio
        tb_1 = lk.InverseKinematicsPoly(np.array([[self.px_init], [-self.H_st]]))
        tb_2 = lk.InverseKinematicsPoly(np.array([[self.mx_init], [-self.H_st]]))
        init_A_tb = tb_1
        init_B_tb = tb_1
        init_C_tb = tb_2
        init_D_tb = tb_2
        FSM = FiniteStateMachine(self.sbrio_freq)
        CORGI = Corgi(self.sbrio_freq, FSM)
        CORGI.step_length = self.L
        CORGI.stance_height = self.H_st
        CORGI.swing_time = self.T_sw
        CORGI.weight_s = self.C_s
        CORGI.weight_u = self.C_u
        CORGI.total_cycle = self.generate_cycle
        CORGI.total_time = self.total_t
        CORGI.setInitPhase(init_A_tb, init_B_tb, init_C_tb, init_D_tb)
        CORGI.move(swing_profile=opt_bez)
        CORGI.exportCSV(csv_filepath)
        print("sbrio exported to", csv_filepath)

        # Export CSV for webot
        tb_1 = lk.InverseKinematicsPoly(np.array([[self.px_init], [-self.H_st]]))
        tb_2 = lk.InverseKinematicsPoly(np.array([[self.mx_init], [-self.H_st]]))
        init_A_tb = tb_1
        init_B_tb = tb_1
        init_C_tb = tb_2
        init_D_tb = tb_2
        FSM = FiniteStateMachine(40)
        CORGI = Corgi(40, FSM)
        CORGI.step_length = self.L
        CORGI.stance_height = self.H_st
        CORGI.swing_time = self.T_sw
        CORGI.weight_s = self.C_s
        CORGI.weight_u = self.C_u
        CORGI.total_cycle = self.generate_cycle
        CORGI.total_time = self.total_t
        CORGI.setInitPhase(init_A_tb, init_B_tb, init_C_tb, init_D_tb)
        CORGI.move(swing_profile=opt_bez)
        CORGI.exportCSV(webot_filepath)
        print("webot exported", webot_filepath)

    def evoCallback(self):
        print("...")


if __name__ == "__main__":
    opt = corgiOptimize()
    # opt.getLiftoffState()
    # opt.run_minimize()
    # # opt.run_evolution()
    # # opt.run_GA()
    # opt.saveData()
    bez_prof_init = np.array(
        [
            [0.04, 0.01, 0.2, 0.05, 0.2, 0.05],
            [0.04, 0.01, 0.2, 0.05, 0.2, 0.05],
            [0.04, 0.01, 0.2, 0.05, 0.2, 0.05],
            [0.04, 0.01, 0.2, 0.05, 0.2, 0.05],
        ]
    )
    bp = np.array(
        [
            0.04002311,
            0.00303396,
            0.01000181,
            -0.01986243,
            0.01305118,
            -0.03105092,
            0.04006933,
            0.00124093,
            0.01742723,
            -0.04600008,
            0.01166539,
            -0.02441404,
            0.04008184,
            0.00328895,
            0.01020455,
            -0.01692465,
            0.01747139,
            -0.04729206,
            0.04011618,
            0.00043849,
            0.01602226,
            -0.04007484,
            0.01082458,
            -0.02170281,
        ]
    )

    opt.plotResult(bp)

    # opt.innerWsPFConstraint(opt.bp_init_guess)
    # opt.outerWsPFConstraint(opt.bp_init_guess)

    print("---")
    # print(opt.opt_result)
