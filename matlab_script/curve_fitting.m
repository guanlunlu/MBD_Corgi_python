clc;
clear;
% load(".\matfiles\com_fitting.mat");
load("./matfiles/com_ic_profile.mat")

%%
M = 0.654;
a = 0.1;
b = 1-a;
rm = COM_y.*-1;
I_com = I_zz.';

% remove first discontinuous data
theta = theta(2:end, :);
rm = rm(2:end, :);
I_com = I_com(2:end, :);
I_hip = I_com + M * rm.^2;

%%
rm_coeff = flip(polyfit(theta, rm, 4))
Icom_coeff = flip(polyfit(theta, I_com, 4))
Ihip_coeff = flip(polyfit(theta, I_hip, 4))

rm_fit = rm_coeff(5)*theta.^4 + rm_coeff(4)*theta.^3 + rm_coeff(3)*theta.^2 + rm_coeff(2)*theta + rm_coeff(1);
Icom_fit = Icom_coeff(5)*theta.^4 + Icom_coeff(4)*theta.^3 + Icom_coeff(3)*theta.^2 + Icom_coeff(2)*theta + Icom_coeff(1);
Ihip_fit = Ihip_coeff(5)*theta.^4 + Ihip_coeff(4)*theta.^3 + Ihip_coeff(3)*theta.^2 + Ihip_coeff(2)*theta + Ihip_coeff(1);
drm_fit = 4*rm_coeff(5)*theta.^3 + 3*rm_coeff(4)*theta.^2 + 2*rm_coeff(3)*theta + rm_coeff(2);

figure;
plot(theta, rm);
hold on;
plot(theta, rm_fit);
hold on;
plot(theta, drm_fit);
hold off

figure;
plot(theta, I_com);
hold on;
plot(theta, Icom_fit);
hold off

figure;
plot(theta, I_hip);
hold on;
plot(theta, Ihip_fit);
hold off;