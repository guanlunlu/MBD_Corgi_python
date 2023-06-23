clc
clear

run poly_fit.m;

load("L1L2_table_0529.mat")
syms t phi_R(t) phi_L(t) theta(t) beta(t) tau_theta(t) tau_beta(t)
syms rm(t) Ic(t)
syms A1 A2 A3 A4
syms B1 B2 B3 B4

% rm = A1 * theta^3 + A2 * theta^2 + A3 * theta + A4;
% Ic = B1 * theta^3 + B2 * theta^2 + B3 * theta + B4;

rm = rm_c(1,1) * theta^3 + rm_c(1,2) * theta^2 + rm_c(1,3) * theta + rm_c(1,4);
Ic = I_zz_c(1,1) * theta^3 + I_zz_c(1,2) * theta^2 + I_zz_c(1,3) * theta + I_zz_c(1,4);

% Ic -> Moment of inertia of C.O.M.
theta_0 = deg2rad(17);

% theta(t) = [1, 0] * (1/2* [1 -1; 1 1]*[phi_R; phi_L] + [1; 0]*theta_0);
% beta(t) = [0, 1] * (1/2* [1 -1; 1 1]*[phi_R; phi_L] + [1; 0]*theta_0);
%%
syms M g
% M = 0.654;
% g = 9.81;

x_m = -1 * rm * sin(beta);
y_m = -1 * rm * cos(beta);

dx_m = diff(x_m, t);
dy_m = diff(y_m, t);

T = 1/2 * M * [dx_m, dy_m] * [dx_m, dy_m].' + 1/2 * Ic * diff(beta, t)^2;
V = M * g * y_m;
simplify(T);
simplify(V);

L = vpa(T-V);
simplify(L);

%%

dL_dtheta = diff(L, theta);
dL_dtheta_dot = diff(L, diff(theta, t));
d_dL_dtheta_dot = diff(dL_dtheta_dot, t);

dL_dbeta = diff(L, beta);
dL_dbeta_dot = diff(L, diff(beta, t));
d_dL_dbeta_dot = diff(dL_dbeta_dot, t);

dL_dtheta = simplify(dL_dtheta);
dL_dbeta = simplify(dL_dbeta);
d_dL_dtheta_dot = simplify(d_dL_dtheta_dot);
d_dL_dbeta_dot = simplify(d_dL_dbeta_dot);

Lag_eqn_theta = simplify(d_dL_dtheta_dot - dL_dtheta == tau_theta);
Lag_eqn_beta = simplify(d_dL_dbeta_dot - dL_dbeta == tau_beta);
%%
% [coeff, term] = coeffs(d_dL_dbeta_dot, [theta diff(theta, t) diff(theta, t, t) beta diff(beta, t) diff(beta, t, t)], 'All')
% eqns = [Lag_eqn_theta Lag_eqn_beta];
% vars = [theta beta];
syms th dth ddth be dbe ddbe Tth Tbe
d_theta = diff(theta, t);
dd_theta = diff(theta, t,t);
d_beta = diff(beta, t);
dd_beta = diff(beta, t,t);

Lag_eqn_theta_subs = subs(Lag_eqn_theta, dd_theta, ddth);
Lag_eqn_theta_subs = subs(Lag_eqn_theta_subs, d_theta, dth);
Lag_eqn_theta_subs = subs(Lag_eqn_theta_subs, theta, th);
Lag_eqn_theta_subs = subs(Lag_eqn_theta_subs, dd_beta, ddbe);
Lag_eqn_theta_subs = subs(Lag_eqn_theta_subs, d_beta, dbe);
Lag_eqn_theta_subs = subs(Lag_eqn_theta_subs, beta, be);
Lag_eqn_theta_subs = subs(Lag_eqn_theta_subs, tau_theta, Tth);
Lag_eqn_theta_subs = subs(Lag_eqn_theta_subs, tau_beta, Tbe);

Lag_eqn_beta_subs = subs(Lag_eqn_beta, dd_theta, ddth);
Lag_eqn_beta_subs = subs(Lag_eqn_beta_subs, d_theta, dth);
Lag_eqn_beta_subs = subs(Lag_eqn_beta_subs, theta, th);
Lag_eqn_beta_subs = subs(Lag_eqn_beta_subs, dd_beta, ddbe);
Lag_eqn_beta_subs = subs(Lag_eqn_beta_subs, d_beta, dbe);
Lag_eqn_beta_subs = subs(Lag_eqn_beta_subs, beta, be);
Lag_eqn_beta_subs = subs(Lag_eqn_beta_subs, tau_theta, Tth);
Lag_eqn_beta_subs = subs(Lag_eqn_beta_subs, tau_beta, Tbe);


eqn_solve = solve(Lag_eqn_theta, Lag_eqn_beta, [th be])
% equationsToMatrix(eqns, vars)

%%
% isolate(Lag_eqn_theta, diff(theta, t, t))

% syms a b y
% [cxy, txy] = coeffs(a*x^2 + b*y, [y x], 'All')
syms x y a
s = solve(x+y == a, y-x==a)







