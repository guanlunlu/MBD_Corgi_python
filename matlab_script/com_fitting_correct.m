clc;
clear;
% load(".\matfiles\com_fitting.mat");
load("./matfiles/com_ic_profile.mat")

%%
clf
M = 0.654;
a = 0.1;
% a = 0.08092;
b = 1-a;

rm = COM_y.*-1;
Ic = I_zz.';

plot(theta, rm,"LineWidth",1)
xlabel("theta [rad]", "FontSize",12)
ylabel("rm [m]","FontSize",12)
set(gca,'fontsize', 12)
set(gcf,'position',[0, 0,800,500])
title("Rm-theta Profile")

figure
plot(theta, Ic,"LineWidth",1)
xlabel("theta [rad]", "FontSize",12)
ylabel("Ic [kg*m^2]","FontSize",12)
set(gca,'fontsize', 12)
set(gcf,'position',[0, 0,800,500])
title("Ic-theta Profile")

%%

q2 = a + a^2/b;
q1 = 2.*a.*rm./b;
q0 = (rm.^2./b) - (Ic./M);

% D = q1.^2 - 4.*q0.*q2;
l = linspace(-1, 1, 100);
l1_parabola = q2 .* l.^2 + q1 .* l + q0;
plot(l, l1_parabola.')
grid on;
%%
L1L2_table = []
for i=1:size(theta)
    rm_ = rm(i);
    Ic_ = Ic(i);
    q2_ = a + a^2/b;
    q1_ = 2.*a.*rm_./b;
    q0_ = (rm_.^2./b) - (Ic_./M);
    D_ = q1_.^2 - 4 * q0_ * q2_;
    if D_ >= 0
        l1_ = -1 * q1_ + sqrt(D_) / (2*q2_);
    else
        l1_ = -1 * q1_ / (2*q2_);
    end
    l2_ = (rm_ + a * l1_)/b;
    L1L2_table = [L1L2_table; [theta(i), l1_, l2_]];
end

plot(theta, L1L2_table(:,2), "LineWidth",1)
hold on
plot(theta, L1L2_table(:,3), "LineWidth",1)
hold off
grid on

legend("l_1", "l_2")
xlabel("theta [rad]", "FontSize",12)
ylabel("[m]","FontSize",12)
set(gca,'fontsize', 12)
set(gcf,'position',[0, 0,800,500])

%%
s = sprintf("Compensate quadratic error: %f", q_eqn(1));
disp(s)
q0 = q0 - q_eqn(1);
q_eqn = q2 .* theta.^2 + q1 .* theta + q0;
plot(q1)
plot(q2 .* theta.^2 + q1 .* theta + q0);



%%
% mass distribution
M = 0.654;  % total mass
a_ls = linspace(0.0,0.99,1000);
b_ls = 1-a_ls;

AB_vec = [a_ls; b_ls];

rm = COM_y .*-1;
Ic = I_zz.';

q2 = a_ls + (a_ls.^2)./b;
q1 = 2 .* a_ls./b_ls .* rm;
q0 = (1./b_ls) .* rm.*rm - Ic/M;

D = q1.^2 - 4.*q0.*q2;
% plot(q2 .* theta.^2 + q1 .* theta + q0);

D_positive = [];
col_idx = []
for col = 1:size(D,2)
    if_D_pos = D(:,col) > 0;
    if sum(if_D_pos) == size(D(:,col), 1)
        D_positive = [D_positive , D(:,col)];
        col_idx = [col_idx, col];
    end
end

plot(D_positive)
a_set = a_ls(col_idx(1));
b_set = b_ls(col_idx(1));
q2_set = a_ls(col_idx(1)) + (a_ls(col_idx(1)).^2)./b
q1_set = 2 .* a_ls(col_idx(1))./b_ls(col_idx(1)) .* rm
q0_set = 1./b_ls(col_idx(1)) .* rm.*rm - Ic/M

D_set = q1_set.^2 - 4.*q0_set.*q2_set;
% plot(D_set)

L_1 = -1 * q1_set + sqrt(D_set) / (2*q2_set);
L_2 = (a_set .* L_1 + rm)./b_set;

figure;
plot(L_1);
hold on;
plot(L_2);
hold on;
plot(rm)
hold off;
title("L1 L2")
legend("L1", "L2", "rm")
%%
out = sim("Copy_of_simplified_model.slx", 30);
%%
simulink_log = load('out.mat');

subplot(2,1,1);
plot(1*(out.simplified_cFy), "LineWidth",1);
hold on;
plot(out.L_Fy + out.R_Fy, "LineWidth",1);
hold off;
title("Joint Y-axis constraint force")
ylabel("Force(N)")
legend("Simplified Model", "Original Model")
xlim([10.8, 12.6]);

subplot(2,1,2);
plot(1*(out.simplified_cFx), "LineWidth",1);
hold on;
plot(out.L_Fx + out.R_Fx, "LineWidth",1);
hold off;
title("Joint X-axis constraint force")
ylabel("Force(N)")
legend("Simplified Model", "Original Model")
xlim([10.8, 12.6]);
%%
LG_y = polyfit(out.simplified_cFy.Data, out.L_Fy.Data + out.R_Fy.Data, 1);
LG_x = polyfit(out.simplified_cFx.Data, out.L_Fx.Data + out.R_Fx.Data, 1);

subplot(2,1,1);
plot(1*(out.simplified_cFy), "LineWidth",1);
hold on;
plot(LG_y(1)*(out.simplified_cFy) + LG_y(2), "LineWidth",1);
hold on;
plot(out.L_Fy + out.R_Fy, "LineWidth",1);
hold on;

title("Joint Y-axis constraint force")
ylabel("Force(N)")
legend("Simplified Model", "Original Model", "Linear regression Simplified Model","Original Model")
xlim([10.8, 12.6]);

subplot(2,1,2);
plot(1*(out.simplified_cFx), "LineWidth",1);
hold on;
plot(LG_x(1)*(out.simplified_cFx)+LG_x(2), "LineWidth",1);
hold on;
plot(out.L_Fx + out.R_Fx, "LineWidth",1);
hold off;
title("Joint X-axis constraint force")
ylabel("Force(N)")
legend("Simplified Model", "Original Model", "Linear regression Simplified Model","Original Model")
xlim([10.8, 12.6]);
%%
save("./matfiles/L1_L2_table", "theta", "L_1", "L_2", "LG_x", "LG_y", "a_set", "b_set")