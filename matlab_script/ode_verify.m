clc;
clear;

filename = "sbrio_data\loadcell\loadcell_data_0519\20230519_sinewave_t_90_45_2_b_0_0_1_22.csv"
data = extractData(filename);
% save("ode_test.mat", "data")
% load("ode_test.mat")
%%
load("L1L2_table_0529.mat");
reply_simulink_phiR = [];
reply_simulink_phiL = [];

for i=1:size(data.rpy_pos_RL, 1)
    reply_sphi_t = getSimulinkPhiRL(data.rpy_pos_RL(i,:).');
    reply_simulink_phiR = [reply_simulink_phiR; reply_sphi_t(1,1)];
    reply_simulink_phiL = [reply_simulink_phiL; reply_sphi_t(2,1)];
end

%%% Run Simulation in Simulink (Use Encoder reply data as input to Simscape model)
secs = seconds(data.t);
simin_phiR = timetable(secs,reply_simulink_phiR);
simin_phiL = timetable(secs,reply_simulink_phiL);
out = sim("simplify_model_fitting_sim.slx", 30);

%%
global theta_list
theta_list = []

% plot(data.t, data.loadcell)
R_torque = reshape(out.R_Torque.Data, [], 1);
L_torque = reshape(out.L_Torque.Data, [], 1);
R_torque = R_torque(1392:end);
L_torque = L_torque(1392:end);
T = out.R_Torque.Time(1392:end);
tau = [T, R_torque, L_torque];

% ODE setup
tspan = [T(1), T(end)];
% initial condition
ini_tb = getThetaBeta(data.rpy_pos_RL(1,:).');
ini_RmIc = getRmIc(ini_tb(1));
ini_Rm = ini_RmIc(1);
ic = [ini_Rm, 0, ini_tb(2), 0];

% opts = odeset('RelTol',1e-2,'AbsTol',1e-4);
[t,X] = ode45(@(t,X) linkleg_ODE(t, X, tau), tspan, ic);

%%
plot(data.t, getThetaBeta((data.rpy_pos_RL).').')
hold on;
plot(theta_list(:,1), theta_list(:,2));
hold off;

%%
% X = [rm, d_rm, beta, d_beta, theta];
% tau = [t, T_phiR, T_phiL]
% theta = [t, theta]
% theta for obtaining Jacobian and rm, Ic, Izz

function dXdt = linkleg_ODE(t, X, tau)
    global theta_list
    m = 0.654;
    g = 9.81;
    rm_coeff = [-0.0132    0.0500    0.0030    0.0110   -0.0035];
    Icom_coeff = [0.0041    0.0043   -0.0013   -0.0001    0.0001];
    Ihip_coeff = [0.0029    0.0097   -0.0103    0.0075   -0.0014];

    rm_ = X(1);
    d_rm_ = X(2);
    beta_ = X(3);
    d_beta_ = X(4);
    
    %%%
    disp("---")
    disp("t = "+ t + ", rm_ = " + rm_ +", d_rm_ = " + d_rm_);
    %%%

    rm_func = @(th) rm_coeff(5)*th.^4 + rm_coeff(4)*th.^3 + rm_coeff(3)*th.^2 + rm_coeff(2)*th + rm_coeff(1) - rm_;
    theta_ = fzero(rm_func, [deg2rad(16.99), deg2rad(160.01)]);

    % Transform torque input to F_rm and T_beta
    tau_R = interp1(tau(:,1), tau(:,2), t, 'linear', 'extrap');
    tau_L = interp1(tau(:,1), tau(:,3), t, 'linear', 'extrap');
    tau_Frb = torqueTransfrom(tau_R, tau_L, theta_, beta_);
    F_rm = tau_Frb(1);
    T_beta = tau_Frb(2);

    %%% debug %%%
    theta_list = [theta_list; [t, theta_]];
    fprintf("theta = %f\n", theta_);
    fprintf("tau_RL = %f, %f\n", tau_R, tau_L);
    fprintf("F_rm, T_beta = %f, %f\n", F_rm, T_beta);
    %%%%%%%%%%%%%
    
%     I_hip = Ihip_coeff(5)*theta.^4 + Ihip_coeff(4)*theta.^3 + Ihip_coeff(3)*theta.^2 + Ihip_coeff(2)*theta + Ihip_coeff(1);

%     d_theta = d_rm_/(4*rm_coeff(5)*theta^3 + 3*rm_coeff(4)*theta^2 + 2*rm_coeff(3)*theta + rm_coeff(2));
%     d_Ic = (4*Icom_coeff(5)*theta^3 + 3*Icom_coeff(4)*theta^2 + 2*Icom_coeff(3)*theta + Icom_coeff(2))*d_theta;

    J_r = [4*rm_coeff(5)*theta_.^3 + 3*rm_coeff(4)*theta_.^2 + 2*rm_coeff(3)*theta_ + rm_coeff(2), 0; 0 1];
    J_Ic = [4*Icom_coeff(5)*theta_^3 + 3*Icom_coeff(4)*theta_^2 + 2*Icom_coeff(3)*theta_ + Icom_coeff(2), 0; 0 1];
    d_Ic = J_Ic * inv(J_r) * [d_rm_; d_beta_];
    d_Ic = d_Ic(1);
    
    I_com = Icom_coeff(5)*theta_.^4 + Icom_coeff(4)*theta_.^3 + Icom_coeff(3)*theta_.^2 + Icom_coeff(2)*theta_ + Icom_coeff(1);
    I_hip = I_com + m*rm_^2;

    dXdt(1) = d_rm_;
    dXdt(2) = (F_rm + m*rm_*d_beta_^2 + m*g*cos(beta_))/m;
    dXdt(3) = d_beta_;
    dXdt(4) = (1/I_hip) *(T_beta - 2*m*rm_*d_rm_*d_beta_ - d_Ic*d_beta_ - m*g*rm_*sin(beta_));
    dXdt = dXdt.';
end
%%


