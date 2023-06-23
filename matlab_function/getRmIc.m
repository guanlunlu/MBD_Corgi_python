function Output = getRmIc(theta)
%GETRMIC Summary of this function goes here
%   Detailed explanation goes here
    rm_coeff = [-0.0132    0.0500    0.0030    0.0110   -0.0035];
    Icom_coeff = [0.0041    0.0043   -0.0013   -0.0001    0.0001];
    Ihip_coeff = [0.0029    0.0097   -0.0103    0.0075   -0.0014];

    Rm = rm_coeff(5)*theta.^4 + rm_coeff(4)*theta.^3 + rm_coeff(3)*theta.^2 + rm_coeff(2)*theta + rm_coeff(1)
    Icom = Icom_coeff(5)*theta.^4 + Icom_coeff(4)*theta.^3 + Icom_coeff(3)*theta.^2 + Icom_coeff(2)*theta + Icom_coeff(1)
    Ihip = Ihip_coeff(5)*theta.^4 + Ihip_coeff(4)*theta.^3 + Ihip_coeff(3)*theta.^2 + Ihip_coeff(2)*theta + Ihip_coeff(1)
    Output = [Rm, Icom, Ihip];
end

