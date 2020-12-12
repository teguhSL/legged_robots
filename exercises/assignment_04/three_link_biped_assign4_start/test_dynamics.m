clc;clc;clc;
clear;
close all;

set_path()

q0 = [pi/7; -pi/7; 0.2];
dq0 = [0.3; 0.2; 3]; 

[x_h, z_h, dx_h, dz_h] = kin_hip(q0, dq0)
[x_swf, z_swf, dx_swf, dz_swf] = kin_swf(q0, dq0)

A_m = eval_A_m(q0)
A_p = eval_A_p(q0)
B = eval_B()
C = eval_C(q0,dq0)
E = eval_energy(q0, dq0)
G = eval_G(q0)
M = eval_M(q0)
[q_p, dq_p] = impact(q0, dq0)

y = [q0; dq0];
parameters = control_hyper_parameters();
dy = eqns(0, y, y, 0, parameters)



options = odeset('RelTol',1e-5, 'Events', @event_func);
h = 0.001; % time step
tmax = 2; % max time that we allow for a single step
tspan = 0:h:tmax;
y0 = [q0; dq0];
t0 = 0;

% we define the solution as a structure to simplify the post-analyses and
% animation, we initialize it to null
sln.T = {};
sln.Y = {};
sln.TE = {};
sln.YE = {};

i = 1;
[T, Y, TE, YE] = ode45(@(t, y) eqns(t, y, y0, i, parameters), t0 + tspan, y0, options);
sln.T{i} = T;
sln.Y{i} = Y;
sln.TE{i} = TE;
sln.YE{i} = YE;



%%%%%%TEST%%%%%%%%%

x0 = [-0.414222; 0.397411; 0.241158; 1.451886; -0.035721; 4.614291; 304.269003; 189.627411; 134.508350; -0.045500; -0.183163];
q0 = x0(1:3);
dq0 = x0(4:6);
x_opt = x0(7:end);
% simulate
num_steps = 1;
sln = solve_eqns(q0, dq0, num_steps, x_opt);
%animate(sln);
results = analyse(sln, x_opt, true);

x0 = [sln.Y{end}(1,:)'; x0(7:end)];
x = x0;
fprintf('s = 0.3 \n params = np.array([%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f])\n', ...
        x(1), x(2), x(3), x(4), x(5), x(6), x(7), x(8), x(9), x(10), x(11))

fprintf('x0 = [%f; %f; %f; %f; %f; %f; %f; %f; %f; %f; %f];\n', ...
        x(1), x(2), x(3), x(4), x(5), x(6), x(7), x(8), x(9), x(10), x(11))
