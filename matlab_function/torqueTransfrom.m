function output = torqueTransfrom(T_phiR, T_phiL, theta, beta)
%TORQUETRANSFROM
%   Transform phi_R phi_L torque to F_rm tau_beta;

rm_coeff = [-0.0132    0.0500    0.0030    0.0110   -0.0035];
Icom_coeff = [0.0041    0.0043   -0.0013   -0.0001    0.0001];
Ihip_coeff = [0.0029    0.0097   -0.0103    0.0075   -0.0014];

J1 = [1/2 -1/2;
      1/2 1/2];

J2 = [4*rm_coeff(5)*theta.^3 + 3*rm_coeff(4)*theta.^2 + 2*rm_coeff(3)*theta + rm_coeff(2), 0; 
      0, 1];

tau_rb = inv(J2.') * inv(J1.') * [T_phiR; T_phiL];

% phiRL = getPhiRL([theta; beta]);
% tau_rb = inv(tauJacobian(rm_coeff, phiRL(1), phiRL(2)).') * [T_phiR; T_phiL];

F_rm = tau_rb(1);
T_beta = tau_rb(2);
output = [F_rm, T_beta];
end

