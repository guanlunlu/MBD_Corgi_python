function [l1, l2] = simplifiedModel(theta)
%   Detailed explanation goes here
a = 0.9;
b = 1-a;
M = 0.654;

A = [0.0107   -0.0481   -0.0135    0.0040];
rm = A(1) * theta.*theta.*theta + A(2) * theta.*theta + A(3) * theta + A(4);
rm = rm * -1;

B = [0.0003   -0.0022    0.0050    0.0];
Ic = B(1) * theta.*theta.*theta + B(2) * theta.*theta + B(3) * theta + B(4);

% l1_ quadratic equation
q2 = a + b^2/a^2;
q1 = 2 * b^2/a^2 * rm;
q0 = b/a^2 * rm.*rm - Ic/M;

l1 = (-q1+sqrt(q1.^2 - 4.*q0.*q2))/(2*q2);
l2 = (b*l1+rm)/a;
end

