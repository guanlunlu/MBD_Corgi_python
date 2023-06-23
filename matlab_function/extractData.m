function [data] = extractDATA(filePath)
%EXTRACTDATA Summary of this function goes here
% Data process for OneLeg_.vi

Traj.time = 10;
Traj.period = 2.5 * 10^-3;
Traj.setup_time = 2;
Traj.wait_time = 1;
KT = 2.2;

raw_exp_data = csvread(filePath);
disp("CSV Data Loaded")

% Trim Data
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
data.loadcell = trim_data(:,59) * 9.80665;
data.rpy_pos_RL = [trim_data(:, 14), trim_data(:, 17)];
data.rpy_vel_RL = [trim_data(:, 15), trim_data(:, 18)];
data.cmd_pos_RL = [trim_data(:, 2), trim_data(:, 7)];
data.rpy_trq_RL = KT .* [trim_data(:, 16), trim_data(:, 19)]
data.t = (Traj.period * (0 : size(data.rpy_pos_RL, 1) - 1)).'
end
