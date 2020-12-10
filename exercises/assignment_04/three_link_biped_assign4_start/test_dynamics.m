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
