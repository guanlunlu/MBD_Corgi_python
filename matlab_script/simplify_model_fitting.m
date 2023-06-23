clc
close all
clear
%%
load("L1L2_table_0529.mat");

% second
Traj.time = 10;
Traj.period = 2.5 * 10^-3;
Traj.setup_time = 2;
Traj.wait_time = 1;
% filepath
filePath = ".\sbrio_data\loadcell\loadcell_data_0519\";
matfilePath = ".\matfiles\";
filename = "20230519_20230505_Wheel-like-walking_h150_v_350_D_1500_T_ratio_1_Tshift_80__13"

raw_exp_data = csvread(filePath + filename + ".csv");

%% Trim Data
full_exp_data = [];
trim_data = [];

for i = 1:size(raw_exp_data,1)
    if raw_exp_data(i,1) ~= 0
        full_exp_data = [full_exp_data; raw_exp_data(i,:)];
    end
end

for i = 1:size(full_exp_data,1)
    if i + 1 <= size(full_exp_data,1)
        if full_exp_data(i+1, 2) ~= 0
            trim_data = [trim_data; full_exp_data(i,:)];
        end
    else
        trim_data = [trim_data; full_exp_data(i,:)];
    end
end


% Convert kgw to Newton
trim_data(:,59) = trim_data(:,59) * 9.80665;

reply_pos_RL = [trim_data(:, 14), trim_data(:, 17)];
cmd_pos_RL = [trim_data(:, 2), trim_data(:, 7)];

time = Traj.period * (0 : size(reply_pos_RL, 1) - 1);
time = time';
secs = seconds(time);

reply_simulink_phiR = [];
reply_simulink_phiL = [];
cmd_simulink_phiR = [];
cmd_simulink_phiL = [];

for i=1:size(reply_pos_RL, 1)
    reply_sphi_t = getSimulinkPhiRL(reply_pos_RL(i,:).');
    reply_simulink_phiR = [reply_simulink_phiR; reply_sphi_t(1,1)];
    reply_simulink_phiL = [reply_simulink_phiL; reply_sphi_t(2,1)];

    cmd_sphi_t = getSimulinkPhiRL(cmd_pos_RL(i,:).');
    cmd_simulink_phiR = [cmd_simulink_phiR; cmd_sphi_t(1,1)];
    cmd_simulink_phiL = [cmd_simulink_phiL; cmd_sphi_t(2,1)];
end
%% Run Simulation in Simulink (Use Encoder reply data as input to Simscape model)
simin_phiR = timetable(secs,reply_simulink_phiR);
simin_phiL = timetable(secs,reply_simulink_phiL);

% load("L1_L2_table.mat")

out = sim("simplify_model_fitting_sim.slx", 30);

%%
% simulink_log = load('out.mat');
% 
% subplot(2,1,1);
% plot(1*(out.simplified_cFy), "LineWidth",1);
% hold on;
% plot(out.L_Fy + out.R_Fy, "LineWidth",1);
% hold off;
% title("Joint Y-axis constraint force")
% ylabel("Force(N)")
% legend("Simplified Model", "Original Model")
% xlim([10.8, 12.6]);
% 
% subplot(2,1,2);
% plot(1*(out.simplified_cFx), "LineWidth",1);
% hold on;
% plot(out.L_Fx + out.R_Fx, "LineWidth",1);
% hold off;
% title("Joint X-axis constraint force")
% ylabel("Force(N)")
% legend("Simplified Model", "Original Model")
% xlim([10.8, 12.6]);
%%
% align data
figure;

t_eFy = linspace(1, size(trim_data(:,59),1) + 1/Traj.period, size(trim_data(:,59),1) + 1/Traj.period) * 2500 * 10^-6;

load_Fy = [zeros(1/Traj.period, 1); trim_data(:,59)] - 0.654 * 9.81;
loadcell_Fy = timeseries(load_Fy, t_eFy);
% ts3 = timeseries((1:5)',[0 10 20 30 40]);
% loadcell_Fy = [t_eFy.', trim_data(:,59)];


subplot(2,1,1);
plot(out.simplified_cFy, "LineWidth",1);
hold on;
% plot(LG_y(1)*(out.simplified_cFy) + LG_y(2), "LineWidth",1);
% hold on;
plot(out.L_Fy + out.R_Fy, "LineWidth",1);
hold on;
plot(loadcell_Fy,"LineWidth",1);
hold off;

title("Joint Y-axis constraint force")
ylabel("Force(N)")
legend("Simplified Model", "Original Model","loadcell")

subplot(2,1,2);
plot(1*(out.simplified_cFx), "LineWidth",1);
hold on;
% plot(LG_x(1)*(out.simplified_cFx)+LG_x(2), "LineWidth",1);
% hold on;
plot(out.L_Fx + out.R_Fx, "LineWidth",1);
hold off;
title("Joint X-axis constraint force")
ylabel("Force(N)")
legend("Simplified Model", "Original Model","Original Model")
% xlim([10.8, 12.6]);
%%
figure;

t_eFy = linspace(1, size(trim_data(:,59),1) + 1/Traj.period, size(trim_data(:,59),1) + 1/Traj.period) * 2500 * 10^-6;

load_Fy = [zeros(1/Traj.period, 1); trim_data(:,59)] - 0.654 * 9.81;
loadcell_Fy = timeseries(load_Fy, t_eFy);
% ts3 = timeseries((1:5)',[0 10 20 30 40]);
% loadcell_Fy = [t_eFy.', trim_data(:,59)];

theta_rpy = [];
beta_rpy = [];
for i = 1:size(reply_simulink_phiR, 1)
    tb_ = getThetaBeta([reply_pos_RL(i,:).']);
    theta_rpy = [theta_rpy; tb_(1,1)];
    beta_rpy = [theta_rpy; tb_(2,1)];
end
theta_rpy = [zeros(1/Traj.period, 1); theta_rpy];
% theta_rpy = wrapTo2Pi(theta_rpy);

subplot(2,1,1);
plot(out.simplified_cFy, "LineWidth",1);
hold on;
plot(out.L_Fy + out.R_Fy, "LineWidth",1);
hold on;
plot(loadcell_Fy,"LineWidth",1);
hold on;
plot(t_eFy, theta_rpy);
hold off

title("Joint Y-axis constraint force")
ylabel("Force(N)")
legend("Simplified Model", "Original Model","loadcell","command theta")
set(gca,'fontsize', 12)
xlim([3.5 12])

subplot(2,1,2);
plot(1*(out.simplified_cFx), "LineWidth",1);
hold on;
plot(out.L_Fx + out.R_Fx, "LineWidth",1);
hold off;
title("Joint X-axis constraint force")
ylabel("Force(N)")
legend("Simplified Model", "Original Model","Original Model")
set(gca,'fontsize', 12)
xlim([3.5 12])
%%
% save("./matfiles/L1_L2_table", "theta", "L_1", "L_2", "LG_x", "LG_y")