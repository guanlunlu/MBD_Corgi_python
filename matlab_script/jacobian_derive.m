clc
clear
syms theta(t) beta(t) phiR(t) phiL(t)
syms A4 A3 A2 A1 A0

theta(t) = 1/2 * phiR - 1/2 * phiL + deg2rad(17);
beta(t) = 1/2 * phiR + 1/2 * phiL;

rm = simplify(A4 * theta^4 + A3 * theta^3 + A2 * theta^2 + A1 * theta + A0);

P_rb = [rm; beta];
J = vpa(jacobian(P_rb, [phiR; phiL]))

syms phi_R phi_L
J_subs = subs(J, phiR, phi_R)
J_subs = subs(J_subs , phiL, phi_L)

J_mf = matlabFunction(J_subs)

J_mf(0,1,1,1,1,10,10)
