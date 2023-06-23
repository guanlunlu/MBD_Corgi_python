clc
clear
load("./matfiles/com_ic_profile.mat")

M = 0.654;
rm = COM_y * -1;
rm = rm(2:end, :);
theta = theta(2:end, :);
I_zz = I_zz(:, 2:end).';

d_th = diff(theta);
d_th = d_th(1);
rm_thetaProfile.d_th = d_th;
rm_thetaProfile.theta = theta;
rm_thetaProfile.rm = rm;
rm_thetaProfile.drm_dtheta = diff(rm)./d_th;
rm_thetaProfile.ddrm_dtheta = diff(rm_thetaProfile.drm_dtheta)/d_th;

Ic_thetaProfile.d_th = d_th;
Ic_thetaProfile.I_zz = I_zz;
Ic_thetaProfile.theta = theta;
Ic_thetaProfile.Ic = I_zz - M* rm.^2;
Ic_thetaProfile.dIc_dtheta = diff(Ic_thetaProfile.Ic)./d_th;
Ic_thetaProfile.ddIc_dtheta = diff(Ic_thetaProfile.dIc_dtheta)./d_th;

% plot(Ic_thetaProfile.Ic);
% hold on
% plot(Ic_thetaProfile.dIc_dtheta);
% hold on
% plot(Ic_thetaProfile.ddIc_dtheta);

% plot(rm)
% hold on
% plot(drm_dtheta)
% hold on
% plot(ddrm_dtheta)
% hold off

%%
t_ = linspace(0,10,1000);
sine_ = sin(t_);
d_sine_ = diff(sine_)./diff(t_);
dd_sine_ = diff(d_sine_)/(10/1000);
plot(sine_)
hold on
plot(d_sine_)
hold on
plot(dd_sine_)
hold off

%%
ft = linspace(0,5,25);
f = ft.^2 - ft - 3;

gt = linspace(1,6,25);
g = 3*sin(gt-0.25);

tspan = [1 5];
ic = 1;
opts = odeset('RelTol',1e-2,'AbsTol',1e-4);
[t,y] = ode45(@(t,y) myode(t,y,ft,f,gt,g), tspan, ic, opts);

plot(t,y)

function dydt = myode(t, y)
f = interp1(ft,f,t); % Interpolate the data set (ft,f) at time t
g = interp1(gt,g,t); % Interpolate the data set (gt,g) at time t
dydt = -f.*y + g; % Evaluate ODE at time t
end

% X = [theta, d_theta, beta, d_beta]
% tau_tb = [t, tau_theta, tau_beta]
function dXdt = linkleg_ODE(t, X, tau_tb, rm_prof, Ic_prof)
    m = 0.654;
    g = 9.81;
    th_ = X(1);
    d_th_ = X(2);
    b_ = X(3);
    d_b_ = X(4);

    rm_ = interp1(rm_prof.theta, rm_prof.rm, th_, 'linear', 'extrap');
    drm_ = interp1(rm_prof.theta, rm_prof.drm_dtheta, th_, 'linear', 'extrap');
    ddrm_ = interp1(rm_prof.theta, rm_prof.ddrm_dtheta, th_, 'linear', 'extrap');

    I_zz_ = interp1(Ic_prof.theta ,Ic_prof.I_zz, th_, 'linear', 'extrap');
    d_Ic_ = interp1(Ic_prof.theta ,Ic_prof.dIc_dtheta, th_, 'linear', 'extrap');

    tau_th_ = interp1(tau_tb(:,1), tau_tb(:,2), t, 'linear', 'extrap');
    tau_b_ = interp1(tau_tb(:,1), tau_tb(:,3), t, 'linear', 'extrap');

    dXdt(1) = d_th_;
    dXdt(2) = (tau_th_ + m*rm_*drm_*d_b_^2 + m*drm_*ddrm_*d_th_^2 - (1/2) * d_Ic_ * d_b_^2 + m*g*drm_*cos(b_)) / (m*d_rm^2);
    dXdt(3) = d_b_;
    dXdt(4) = (1/I_zz_) * (tau_b_ - (2*m*rm_*drm_+d_Ic_)*d_th_*d_b_ - m*g*rm_*sin(b_));
end

