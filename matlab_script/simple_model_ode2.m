clc;
clear;

filename = "sbrio_data\loadcell\loadcell_data_0519\20230519_sinewave_t_90_45_1_b_0_0_1_20.csv"
data = extractData(filename);
save("ode_test.mat", "data");
% load("ode_test.mat")

%%

global theta_list
theta_list = []

% plot(data.t, data.loadcell)
tau = [data.t, data.rpy_trq_RL];
% tau = [data.t, zeros([size(data.t, 1) 2])];

% ODE setup
tspan = [data.t(1), data.t(end)];
% tspan = [data.t(1), 1];
% t0 = 0;
% tf = 0.1;
% nsteps = 100;
% tspan = linspace(t0,tf,nsteps);

% initial condition
ini_tb = getThetaBeta(data.rpy_pos_RL(1,:).');
ini_RmIc = getRmIc(ini_tb(1));
ini_Rm = ini_RmIc(1);
ic = [ini_Rm, 0, ini_tb(2), 0];

opts = odeset('RelTol',1e-2,'AbsTol',1e-4);
% options= odeset('Reltol',0.001,'Stats','on');
[t,X] = ode45(@(t,X) linkleg_ODE(t, X, tau), tspan, ic, opts);
% y = ode2(@vdp1,tspan,[2 0]);  
% X = ode2(@(t,X) linkleg_ODE(t, X, tau), tspan, ic);
%%
% checkpoint_filename = "0623_ode.mat";
% save(checkpoint_filename);
%%
plot(data.t, getThetaBeta((data.rpy_pos_RL).').')
hold on;
% plot(theta_list(:,1), theta_list(:,2));
plot(t, X(:,1))
hold off;

% plot(X(:,1))

%%
data.rpy_pos_tb = getThetaBeta((data.rpy_pos_RL).').';
rpy_RmIc = getRmIc(data.rpy_pos_tb(:,1));
plot(t, X(:,1))
hold on
plot(data.t, rpy_RmIc(:,1));
% plot(theta_list(:,1), theta_list(:,2));
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
%     Ihip_coeff = [0.0029    0.0097   -0.0103    0.0075   -0.0014];

    rm_ = X(1);
    d_rm_ = X(2);
    beta_ = X(3);
    d_beta_ = X(4);

    rm_func = @(th) rm_coeff(5)*th.^4 + rm_coeff(4)*th.^3 + rm_coeff(3)*th.^2 + rm_coeff(2)*th + rm_coeff(1) - rm_;
    theta_ = fzero(rm_func, [deg2rad(16.99), deg2rad(160.01)]);
%     r = roots([-0.0132    0.0500    0.0030    0.0110   -0.0035-rm_]);
%     for i=1:4
%         if isreal(r(i))
%             if r(i) > deg2rad(16.99) && r(i) < deg2rad(170.01)
%                theta_ = r(i);
%             end
%         end
%     end

    % Transform torque input to F_rm and T_beta
    tau_R_mtr = interp1(tau(:,1), tau(:,2), t, 'linear', 'extrap');
    tau_L_mtr = interp1(tau(:,1), tau(:,3), t, 'linear', 'extrap');
    % Friction
    J1 = [1/2 -1/2;
          1/2 1/2];
    J2 = [4*rm_coeff(5)*theta_.^3 + 3*rm_coeff(4)*theta_.^2 + 2*rm_coeff(3)*theta_+ rm_coeff(2), 0; 
          0, 1];
    d_phi_RL = inv(J2 * J1) * [d_rm_; d_beta_];
%     tau_R_ff = -0.5 * sign(d_phi_RL(1)) + 0.5 * sign(d_phi_RL(1)) * tau_R_mtr;
%     tau_L_ff = -0.5 * sign(d_phi_RL(2)) + 0.5 * sign(d_phi_RL(2)) * tau_L_mtr;
%     tau_R_ff = - (0.3 + 0.1 * d_phi_RL(1));
%     tau_L_ff = - (0.3 + 0.1 * d_phi_RL(2));
    tau_R_ff = -(0.88 * sign(d_phi_RL(1)) + 0.8 * d_phi_RL(1));
    tau_L_ff = -(0.88 * sign(d_phi_RL(2)) + 0.8 * d_phi_RL(2));
    
    tau_R = tau_R_mtr + tau_R_ff;
    tau_L = tau_L_mtr + tau_L_ff;

    tau_Frb = torqueTransfrom(tau_R, tau_L, theta_, beta_);
    F_rm = tau_Frb(1);
    T_beta = tau_Frb(2);    

    %%% debug %%%
%     disp("---")
%     fprintf("t = %f, rm = %f, d_rm = %f, beta = %f, d_beta = %f \n", t, rm_, d_rm_, beta_, d_beta_);
%     fprintf("theta = %f\n", theta_);
%     fprintf("tau_RL = %f, %f\n", tau_R, tau_L);
%     fprintf("tau_RL_friction = %f, %f\n", tau_R_ff, tau_L_ff);
%     fprintf("d_phi_RL = [%f, %f]\n", d_phi_RL(1), d_phi_RL(2));
%     fprintf("F_rm, T_beta = %f, %f\n", F_rm, T_beta);
%     fprintf("m*rm_*d_beta_^2 = %f\n", m*rm_*d_beta_^2);
%     fprintf("m*g*cos(beta_) = %f\n", m*g*cos(beta_));
%     fprintf("dd_rm = %f\n", (F_rm + m*rm_*d_beta_^2 + m*g*cos(beta_))/m);
    %%%%%%%%%%%%%

    J_r = [4*rm_coeff(5)*theta_.^3 + 3*rm_coeff(4)*theta_.^2 + 2*rm_coeff(3)*theta_ + rm_coeff(2), 0; 0 1];
    J_Ic = [4*Icom_coeff(5)*theta_^3 + 3*Icom_coeff(4)*theta_^2 + 2*Icom_coeff(3)*theta_ + Icom_coeff(2), 0; 0 1];
    d_Ic = J_Ic * inv(J_r) * [d_rm_; d_beta_];
    d_Ic = d_Ic(1);
    
%     fprintf("d_Ic = %f\n", d_Ic);

    I_com = Icom_coeff(5)*theta_.^4 + Icom_coeff(4)*theta_.^3 + Icom_coeff(3)*theta_.^2 + Icom_coeff(2)*theta_ + Icom_coeff(1);
    I_hip = I_com + m*rm_^2;

    dXdt(1) = d_rm_;
    dXdt(2) = (F_rm + m*rm_*d_beta_^2 + m*g*cos(beta_))/m;
    dXdt(3) = d_beta_;
    dXdt(4) = (1/I_hip)*(T_beta - 2*m*rm_*d_rm_*d_beta_ - d_Ic*d_beta_ - m*g*rm_*sin(beta_));
    dXdt = dXdt.';
%     tic
    theta_list(end+1, :) = [t, theta_];
%     toc
end
%%