clc
clf
clear

%% Profile Parameter
ThetaProfile.bias = deg2rad(90);
ThetaProfile.amplitude = deg2rad(45);
ThetaProfile.frequency = 2;

BetaProfile.bias = deg2rad(0);
BetaProfile.amplitude = deg2rad(0);
BetaProfile.frequency = 1;

% second
Traj_time = 10;
Traj_period = 2.5 * 10^-3;
Traj_setup_time = 2;
Traj_wait_time = 1;

% filepath
filepath = "./csv/"

%%
% initial angle
phiRL_0 = getPhiRL([ThetaProfile.bias; BetaProfile.bias]);

setup_traj.len = Traj_setup_time/Traj_period;
setup_traj.time = [linspace(0,Traj_setup_time, setup_traj.len), linspace(Traj_setup_time + Traj_period, Traj_setup_time + Traj_wait_time, Traj_wait_time/Traj_period)];

setup_traj.R = [linspace(0, phiRL_0(1,1), setup_traj.len), linspace(phiRL_0(1,1), phiRL_0(1,1), Traj_wait_time/Traj_period)];
setup_traj.L = [linspace(0, phiRL_0(2,1), setup_traj.len), linspace(phiRL_0(2,1), phiRL_0(2,1), Traj_wait_time/Traj_period)];

setup_traj.theta = linspace(0, 0, setup_traj.len + Traj_wait_time/Traj_period);
setup_traj.beta = linspace(0, 0, setup_traj.len + Traj_wait_time/Traj_period);

setup_traj.df = [setup_traj.time; setup_traj.theta; setup_traj.beta; setup_traj.R; setup_traj.L].';


%%
t = Traj_setup_time + Traj_wait_time + Traj_period;
dataframe = [];

% dataframe = ["time", "theta", "beta", "phi_R", "phi_L"];
while t <= Traj_time + Traj_period  
    theta_t = ThetaProfile.bias + ThetaProfile.amplitude * sin(2*pi*ThetaProfile.frequency * t);
    beta_t = BetaProfile.bias + BetaProfile.amplitude * sin((2*pi*BetaProfile.frequency) * t);
    phiRL = getPhiRL([theta_t; beta_t]);
    
    dataframe = [dataframe; [t, theta_t, beta_t, phiRL(1,1), phiRL(2,1)]];
    t = t + Traj_period;
end

dataframe = [setup_traj.df; dataframe]

[dfrow, dfcol] = size(dataframe);
% plot(dataframe(2:dfrow, 1), dataframe(2:dfrow, 4));
% hold on
% plot(dataframe(2:dfrow,1), dataframe(2:dfrow,5));
% hold off
%% save csv file
sbrio_df = [dataframe(:, 4), dataframe(:, 5)];
[sbrow, sbcol] = size(sbrio_df);
zero_col = linspace(0,0,sbrow).';
sbrio_df = [sbrio_df, zero_col, zero_col, zero_col, zero_col, zero_col, zero_col, zero_col, zero_col, zero_col, zero_col];

filename = "sinewave_t_" + num2str(rad2deg(ThetaProfile.bias)) + "_" + num2str(rad2deg(ThetaProfile.amplitude)) + "_" + num2str(ThetaProfile.frequency) ...
    + "_b_" + num2str(rad2deg(BetaProfile.bias)) + "_" + num2str(rad2deg(BetaProfile.amplitude)) + "_" + num2str(BetaProfile.frequency) + ".csv"

csvwrite(filepath + filename, sbrio_df)

%%
sampleTime = Traj_period;
numSteps = Traj_time/Traj_period;

time = sampleTime*(0:numSteps-1);
time = time';
secs = seconds(time);

simulink_phiR = [];
simulink_phiL = [];

for i=1:dfrow
    phi_t = dataframe(i, 4:5).';
    sphi_t = getSimulinkPhiRL(phi_t);
    simulink_phiR = [simulink_phiR; sphi_t(1,1)];
    simulink_phiL = [simulink_phiL; sphi_t(2,1)];
end

simin_phiR = timetable(secs,simulink_phiR);
simin_phiL = timetable(secs,simulink_phiL);



