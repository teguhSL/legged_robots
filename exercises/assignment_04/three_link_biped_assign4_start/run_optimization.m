clc;
clear;
close all;

%% optimize
% optimize the initial conditions and controller hyper parameters
q0 = [pi/9; -pi/9; 0];
dq0 = [0; 0; 8]; 
x0 = [q0; dq0; control_hyper_parameters()];

% use fminsearch and optimset to control the MaxIter
%options = optimset('Display','iter','PlotFcns',@optimplotfval);
options = optimset('PlotFcns',@optimplotfval);
x = fminsearch(@optimziation_fun, x0, options);

%% simulate solution

% extract parameters
q0 = x(1:3);
dq0 = x(4:6);
x_opt = x(7:end);

% simulate
num_steps = 10;
sln = solve_eqns(q0, dq0, num_steps, x_opt);
animate(sln);
results = analyse(sln, x_opt, true);


% %test
% tic()
% for i=1:100
% sln = solve_eqns(q0, dq0, 2, x_opt);
% end
% t = toc()
% disp(t)
% 
% tic()
% for i=1:1000
%     dy = eqns(0, x0, x0, num_steps, x_opt);
% end
% t = toc()
% disp(t)
% 
% tic()
% for i=1:10000
%     eval_G(q0);
% end
% t = toc()
% disp(t/10)
% 
tic()
for i=1:1000000
    sin(20.3);
end
t = toc()
disp(t)