function J = tauJacobian(rm_coeff, phi_R, phi_L)
%TAUJACOBIAN Summary of this function goes here
%   Detailed explanation goes here
A4 = rm_coeff(5);
A3 = rm_coeff(4);
A2 = rm_coeff(3);
A1 = rm_coeff(2);
A0 = rm_coeff(1);
J = reshape([A1.*5.0e-1+A2.*(phi_L.*-5.0e-1+phi_R.*5.0e-1+2.96705972839036e-1)+A3.*(phi_L.*-5.0e-1+phi_R.*5.0e-1+2.96705972839036e-1).^2.*1.5+A4.*(phi_L.*-5.0e-1+phi_R.*5.0e-1+2.96705972839036e-1).^3.*2.0,5.0e-1,A1.*-5.0e-1-A2.*(phi_L.*-5.0e-1+phi_R.*5.0e-1+2.96705972839036e-1).*1.0-A3.*(phi_L.*-5.0e-1+phi_R.*5.0e-1+2.96705972839036e-1).^2.*1.5-A4.*(phi_L.*-5.0e-1+phi_R.*5.0e-1+2.96705972839036e-1).^3.*2.0,5.0e-1],[2,2]);
end

