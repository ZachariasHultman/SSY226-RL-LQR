clear all; clc; close all

Ac = [0 -1; 0 -0.1];
Bc = [0; 1];
n = size(Ac,2);
m = size(Bc,2);
C = eye(n);
D = zeros(n,m);

M = eye(n);
R = 10*eye(m);

iter = 1e4;%1e5;
h = 0.01;%1e-5;

%Discretize system
A = expm(Ac);
B = pinv(Ac)*(A-eye(n))*Bc;

% %Undiscretized system
% A = Ac;
% B = Bc;

%Stationary solution to Ricatti equation:

% [X,Ke,L] = icare(A,B,M,R,[],[]);
[X,Ke,L] = idare(A,B,M,R,[],[]);
Ke = [-1 2];

x_offline = zeros(n,iter);
u_offline = zeros(m,iter);
t = get_time(h, iter);%linspace(0,100,iter);
x_offline(:,1) = [1; 0];

for i = 2:iter
    [noise,amplitude,freq, phase] = GenerateNoise(t(i));
    u_offline(:,i) = -Ke*x_offline(:,i-1) + noise/1;
    x_offline(:,i) = A * x_offline(:,i-1) + B*u_offline(:,i);
    all_amp(:,i) = amplitude;
    all_freq(:,i) = freq;
    all_phase(:,i) = phase;
end

%%Plot values
plot(t,u_offline)
hold on
plot(t,x_offline)
hold off
legend("u","x_1","x_2")
% TT = [t', x_offline', u_offline'];
save('offlinedata.mat','x_offline', 'u_offline', 'all_amp', 'all_freq', 'all_phase', 'Ke', 'A', 'B', 't')
% FileData = load('offlinedata.mat');
% % csvwrite('offlinedata1.csv', FileData.M);
% 
% vars = {'t','x1', 'x2','u', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10',...
%     'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20',...
%     'a21', 'a22', 'a23', 'a24',...
%     'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10',...
%     'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20',...
%     'f21', 'f22', 'f23', 'f24',...
%     'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10',...
%     'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p20',...
%     'p21', 'p22', 'p23', 'p24'};
% TT = [t', x_offline', u_offline', all_amp', all_freq', all_phase'];
% T = table(TT(:,1), TT(:,2), TT(:,3), TT(:,4), TT(:,5), TT(:,6),TT(:,7),TT(:,8),TT(:,9),TT(:,10),...
%     TT(:,11), TT(:,12), TT(:,13), TT(:,14), TT(:,15), TT(:,16),TT(:,17),TT(:,18),TT(:,19),TT(:,20),...
%     TT(:,21), TT(:,22), TT(:,23), TT(:,24), TT(:,25), TT(:,26),TT(:,27),TT(:,28),TT(:,29),TT(:,30),...
%     TT(:,31), TT(:,32), TT(:,33), TT(:,34), TT(:,35), TT(:,36),TT(:,37),TT(:,38),TT(:,39),TT(:,40),...
%     TT(:,41), TT(:,42), TT(:,43), TT(:,44), TT(:,45), TT(:,46),TT(:,47),TT(:,48),TT(:,49),TT(:,50),...
%     TT(:,51), TT(:,52), TT(:,53), TT(:,54), TT(:,55), TT(:,56),TT(:,57),TT(:,58),TT(:,59),TT(:,60),...
%     TT(:,61), TT(:,62), TT(:,63), TT(:,64), TT(:,65), TT(:,66),TT(:,67),TT(:,68),TT(:,69),TT(:,70),...
%     TT(:,71), TT(:,72), TT(:,73), TT(:,74), TT(:,75), TT(:,76),...
%     'VariableNames',vars);
% % disp(T)
% writetable(T,"offlinedata.csv")
% 
% for i = 1:24
%    f(i) = append("f",num2str(i)); 
% end



function t = get_time(time_interval, iter)

t = zeros(1,iter);

for i = 1:iter
    t(i) = (i-1)*time_interval;
end
end