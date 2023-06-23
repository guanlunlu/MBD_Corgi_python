clc;
clear;
close all;
filename = "sbrio_data\loadcell\loadcell_data_0519\20230519_20230505_Wheel-like-walking_h150_v_450_D_1500_T_ratio_1_Tshift_80__8.csv"
data = extractData(filename);
save("ode_test.mat", "data");
%%
data.rpy_pos_tb = [];
data.rpy_vel_tb = [];
for i = 1:size(data.rpy_pos_RL, 1)
    data.rpy_pos_tb = [data.rpy_pos_tb; getThetaBeta(data.rpy_pos_RL(i, :).').'];
    J_tb = [1/2 -1/2; 1/2 1/2];
    vel_tb = [1/2 -1/2; 1/2 1/2]*(data.rpy_vel_RL(i,:).');
    data.rpy_vel_tb = [data.rpy_vel_tb; vel_tb.'];
end

interval = data.t(2)-data.t(1);
% data.rpy_vel_tb = diff(data.rpy_pos_tb)/interval;
data.rpy_vel_tb = smoothdata(data.rpy_vel_tb, "gaussian", 5);
data.rpy_acc_tb = smoothdata(diff(data.rpy_vel_tb)/interval, "gaussian", 5);

plot(data.rpy_pos_tb);
hold on;
plot(data.rpy_vel_tb);
hold on;
plot(data.rpy_acc_tb);
hold off;
%%
Tau_rb = [];
for i = 1:size(data.rpy_acc_tb, 1)
    theta = data.rpy_pos_tb(i, 1);
    beta = data.rpy_pos_tb(i, 2);
    dtheta = data.rpy_vel_tb(i, 1);
    dbeta = data.rpy_vel_tb(i, 2);
    ddtheta = data.rpy_acc_tb(i, 1);
    ddbeta = data.rpy_acc_tb(i, 2);
    Tau_rb = [Tau_rb; inverseLegDynamic(theta, dtheta, ddtheta, beta, dbeta, ddbeta)];
end
%%
plot(Tau_rb(:,1))
hold on
plot(data.loadcell);
hold off
%%
torque_inertia_RL = [];
for i=1:size(Tau_rb, 1)
    rm_coeff = [-0.0132    0.0500    0.0030    0.0110   -0.0035];
    Icom_coeff = [0.0041    0.0043   -0.0013   -0.0001    0.0001];
    Ihip_coeff = [0.0029    0.0097   -0.0103    0.0075   -0.0014];

    J1 = [1/2 -1/2;
          1/2 1/2];
    
    J2 = [4*rm_coeff(5)*theta.^3 + 3*rm_coeff(4)*theta.^2 + 2*rm_coeff(3)*theta + rm_coeff(2), 0; 
          0, 1];

    tau_phi = J1.' * J2.' * Tau_rb(i, :).';
    torque_inertia_RL = [torque_inertia_RL; tau_phi.'];
end

figure
plot(data.t(1:end-1), torque_inertia_RL(:,1))
hold on
plot(data.t, smoothdata(data.rpy_trq_RL(:,1)))
hold off
legend("inertia estimated", "measure");

figure
plot(data.t(1:end-1), torque_inertia_RL(:,2))
hold on
plot(data.t, smoothdata(data.rpy_trq_RL(:,2)))
hold off
legend("inertia estimated", "measure");

%%
function Tau_rb = inverseLegDynamic(theta, dtheta, ddtheta, beta, dbeta, ddbeta)
    m = 0.654;
    g = 9.81;
    rm_coeff = [-0.0132    0.0500    0.0030    0.0110   -0.0035];
    Icom_coeff = [0.0041    0.0043   -0.0013   -0.0001    0.0001];

    rm_ = rm_coeff(5) * theta^4 + rm_coeff(4) * theta^3 + rm_coeff(3) * theta^2 + rm_coeff(2) * theta + rm_coeff(1);
    drm_ = (4*rm_coeff(5)*theta^3 + 3*rm_coeff(4)*theta^2 + 2*rm_coeff(3)*theta + rm_coeff(2)) * dtheta;
    ddrm_ = (4*rm_coeff(5)*theta^3 + 3*rm_coeff(4)*theta^2 + 2*rm_coeff(3)*theta + rm_coeff(2)) * ddtheta + (12*rm_coeff(5)*theta^2 + 6*rm_coeff(4)*theta + 2*rm_coeff(3)) * dtheta^2;

    Ic = Icom_coeff(5) * theta^4 + Icom_coeff(4) * theta^3 + Icom_coeff(3) * theta^2 + Icom_coeff(2) * theta + Icom_coeff(1);
    dIc_ = (4*Icom_coeff(5)*theta^3 + 3*Icom_coeff(4)*theta^2 + 2*Icom_coeff(3)*theta + Icom_coeff(2)) * dtheta;
    
    F_rm = m*ddrm_ - m*rm_*dbeta^2 - m*g*cos(beta);
    T_beta = (Ic + m*rm_^2)*ddbeta + 2*m*rm_*drm_*dbeta + dIc_*dbeta + m*g*rm_*sin(beta);
    Tau_rb = [F_rm, T_beta];
end