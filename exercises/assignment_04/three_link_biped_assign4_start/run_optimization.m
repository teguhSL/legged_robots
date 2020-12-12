clc;
clear;
close all;

%% optimize
% optimize the initial conditions and controller hyper parameters
q0 = [pi/9; -pi/9; 0.3];
dq0 = [0; 0; 8]; 
x0 = [q0; dq0; control_hyper_parameters()];
%x0 = [-0.236131; 0.307839; 0.137983; 0.671299; -1.113151; 4.499133; 687.358046; 136.697506; 82.750343; -0.408726; 0.121647];
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
%animate(sln);
results = analyse(sln, x_opt, true);

disp('The optimal parameters are:')
fprintf('s = 0.3 \n params = np.array([%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f])\n', ...
        x(1), x(2), x(3), x(4), x(5), x(6), x(7), x(8), x(9), x(10), x(11))

fprintf('x0 = [%f; %f; %f; %f; %f; %f; %f; %f; %f; %f; %f];\n', ...
        x(1), x(2), x(3), x(4), x(5), x(6), x(7), x(8), x(9), x(10), x(11))
