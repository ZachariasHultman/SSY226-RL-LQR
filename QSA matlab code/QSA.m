clear all; clc; close all
addpath('Offline data generator')

load('offlinedata.mat')

%Offline data plot
plot(t,u_offline)
hold on
plot(t,x_offline)
hold off
legend("u","x_1","x_2")

% n = size(A,2);
[n, m] = size(B);

K_N = zeros(size(1,n));
M = eye(n);
R = 10*eye(m);

x_init = [1, 0]';
u_init = -K*x_init + GenerateNoise(t(1));

U = [x_init; u_init];

s = 0.5*(length(U)*(length(U)+1));

theta = zeros(s,1);

dtheta = zeros(s,1);
phi = 0*u_init;

g = 1.5;

iter = 1;

theta_record = [];
dtheta_record = [];
phi_record = [];
t_record = [0];
%============Start of while loop==========================
count = 1

while (count<100)
    x_off = x_offline(:,iter);
    u_off = u_offline(:,iter);
    t_off = t(:,iter);

    xdot = x_offline(:,iter+1);
    udot = u_offline(:,iter+1);

    c_xu = cost_func(x_off,u_off, M, R);
    d_xu = cost_func(x_off,u_off, M, R);
    d_xphi = cost_func(x_off,phi, M, R);

    U1 = [x_off; u_off];
    U2 = [x_off; phi];
    Udot = [xdot; udot];

    Si_U1 = GetKron(U1, n, m);
    Si_U2 = GetKron(U2, n, m);

    %Eq. 23:
    Q = Q_func(d_xu, theta, Si_U1);

    %Eq. 24:

    %Zeta(t):
    zeta(:,iter) = Si_U2 - Si_U1 + diff_Si_func(U2, Udot, n, m);

    %b(t):
    b(:,iter) = c_xu - d_xu + d_xphi + cost_diff_func(x_off, xdot, u_off, udot, M, R);

    %Theta 
    a = g/ (t_off + 1)

    dtheta  = -a * (zeta(:,iter)'*theta + b(:,iter))*zeta(:,iter);

    theta = theta + dtheta;
    
    Q_phi = GetVec2mat(theta,n , m);
    Qux = Q_phi(end,1:n);
    
    phi = -inv(R)*Qux*x_off;
    
    iter = iter + 1;
    count = count + 1;
    
    %Store theta and phi
    theta_record = [theta_record theta];
    dtheta_record = [dtheta_record dtheta];
    phi_record = [phi_record phi];
    t_record = [t_record t_off];
end
%============End of while loop============================

K_N = Qux;
check_val(A,B,x_init, u_init, iter, K_N, n, m, t_record)

%Plots
figure()
plot([1:iter-1],phi_record)
title("Phi record")

figure()
plot([1:iter-1],theta_record)
title("\theta record")

figure()
plot([1:iter-1],dtheta_record)
title("\frac{d\theta}{dt} record")
title('$\displaystyle\frac{d\theta}{dt}$','interpreter','latex')