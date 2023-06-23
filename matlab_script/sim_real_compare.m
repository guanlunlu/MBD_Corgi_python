clear
clc
load('sim_sinewave_t_90_45_2_b_0_0_1.mat')

%%
csvPath = ".\sbrio_data\loadcell\20230517_sinewave_t_90_45_2_b_0_0_1_9.csv"
raw_exp_data = csvread(csvPath);
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
% figure;
% r_cmd = []
% for i = 1:size(logsout{16}.Values.Data, 1)
%     r_cmd = [r_cmd; deg2rad(90-17) - logsout{16}.Values.Data(1,1,i)];
% end

plot(trim_data(:,2))
hold on;
plot(deg2rad(90-17) - logsout{16}.Values.Data)
hold off;
legend('trim_data','r_cmd');
% plot(trim_data(:,59))

%%
simulation_Fy = [];
R_Fx = [];
R_Fy = [];
L_Fx = [];
L_Fy = [];

for i = 1:size(logsout{10}.Values.Data, 3)
    R_Fx = [R_Fx; logsout{10}.Values.Data(1,1,i)];
    R_Fy = [R_Fy; logsout{10}.Values.Data(2,1,i)];
    L_Fx = [L_Fx; logsout{4}.Values.Data(1,1,i)];
    L_Fy = [L_Fy; logsout{4}.Values.Data(2,1,i)];
end

F_x = R_Fx + L_Fx;
F_y = R_Fy + L_Fy;
F_y = F_y + 6.48952;

t_Fy = linspace(1,size(F_y,1),size(F_y,1)) * 2500 * 10^-6;
t_eFy = linspace(1,size(trim_data(:,59),1), size(trim_data(:,59),1)) * 2500 * 10^-6;
plot(t_Fy, F_y,'LineWidth',1);
hold on;
plot(t_eFy, trim_data(:,59)-0.2, 'LineWidth',1);
hold off;

xlabel("Time (s)")
ylabel("Force (N)")
title('sim_sinewave_t_90_45_2_b_0_0_1','Interpreter', 'none')
legend('Simscape Model','Experiment Data');

