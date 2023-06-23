clc
clear
%%
% second
Traj.time = 10;
Traj.period = 2.5 * 10^-3;
Traj.setup_time = 2;
Traj.wait_time = 1;

% filepath
filePath = ".\sbrio_data\loadcell\loadcell_data_0519\";
matfilePath = ".\matfiles\";
filename = "20230519_20230505_Wheel-like-walking_h150_v_450_D_1500_T_ratio_1_Tshift_80__8"

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

%%
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
simout_1 = sim('linkleg_trajectory', 30);
reply_matfile_ = matfilePath + "reply_" + filename + ".mat";
movefile('.\out.mat', reply_matfile_);

%% Run Simulation in Simulink (Use Command data as input to Simscape model)
cmd_simulink_phiR = unwrap(cmd_simulink_phiR);
cmd_simulink_phiL = unwrap(cmd_simulink_phiL);
simin_phiR = timetable(secs, cmd_simulink_phiR);
simin_phiL = timetable(secs, cmd_simulink_phiL);

simout_2 = sim('linkleg_trajectory', 30);
cmd_matfile_ = matfilePath + "cmd_" + filename + ".mat";
movefile('.\out.mat', cmd_matfile_);

%% Analyze Data
rpy_data = load(reply_matfile_);
cmd_data = load(cmd_matfile_);

rpy_data.R_Fx = [];
rpy_data.R_Fy = [];
rpy_data.L_Fx = [];
rpy_data.L_Fy = [];
cmd_data.R_Fx = [];
cmd_data.R_Fy = [];
cmd_data.L_Fx = [];
cmd_data.L_Fy = [];


for i = 1:size(logsout{10}.Values.Data, 3)
    rpy_data.R_Fx = [rpy_data.R_Fx; rpy_data.logsout{11}.Values.Data(1,1,i)];
    rpy_data.R_Fy = [rpy_data.R_Fy; rpy_data.logsout{11}.Values.Data(2,1,i)];
    rpy_data.L_Fx = [rpy_data.L_Fx; rpy_data.logsout{5}.Values.Data(1,1,i)];
    rpy_data.L_Fy = [rpy_data.L_Fy; rpy_data.logsout{5}.Values.Data(2,1,i)];
    cmd_data.R_Fx = [cmd_data.R_Fx; cmd_data.logsout{11}.Values.Data(1,1,i)];
    cmd_data.R_Fy = [cmd_data.R_Fy; cmd_data.logsout{11}.Values.Data(2,1,i)];
    cmd_data.L_Fx = [cmd_data.L_Fx; cmd_data.logsout{5}.Values.Data(1,1,i)];
    cmd_data.L_Fy = [cmd_data.L_Fy; cmd_data.logsout{5}.Values.Data(2,1,i)];
end

rpy_data.F_x = rpy_data.R_Fx + rpy_data.L_Fx;
rpy_data.F_y = rpy_data.R_Fy + rpy_data.L_Fy;
rpy_data.F_y = rpy_data.F_y + 6.48952;

cmd_data.F_x = cmd_data.R_Fx + cmd_data.L_Fx;
cmd_data.F_y = cmd_data.R_Fy + cmd_data.L_Fy;
cmd_data.F_y = cmd_data.F_y + 6.48952;

% Eliminate simscape delay due to initialize
start_row = 1/Traj.period;
rpy_data.F_x = rpy_data.F_x(start_row:end, 1);
rpy_data.F_y = rpy_data.F_y(start_row:end, 1);
cmd_data.F_x = cmd_data.F_x(start_row:end, 1);
cmd_data.F_y = cmd_data.F_y(start_row:end, 1);

t_Fy = linspace(1,size(rpy_data.F_y,1),size(rpy_data.F_y,1)) * 2500 * 10^-6;
t_eFy = linspace(1,size(trim_data(:,59),1), size(trim_data(:,59),1)) * 2500 * 10^-6;

figure;
plot(t_Fy, rpy_data.F_y,'LineWidth',1);
hold on;
plot(t_Fy, cmd_data.F_y,'LineWidth',1);
hold on;
plot(t_eFy, trim_data(:,59), 'LineWidth',1);
hold off;

xlabel("Time (s)")
ylabel("Force (N)")
title(filename,'Interpreter', 'none')
legend('Simscape Model (Reply)','Simscape Model (CMD)','Experiment Data');