% run the following code
clear all
load('sln_test.mat') 
skip = 5;
Y_size = size(sln_test.Y);
num_steps = Y_size(2);% total number of steps the robot has taken (find this from sln)
x_foots = [];
z_foots = [];

for j = 1:num_steps
    Y = sln_test.Y{j};%
    [N, ~] = size(Y);
    for i = 1:skip:N % what does skip do?
        q = Y(i,1:3); dq = Y(i,4:6);%
        [x_swf, z_swf, dx_swf, dz_swf] = kin_swf(q, dq);
        x_foots = [x_foots; x_swf];
        z_foots = [z_foots; z_swf];
    end
end

figure()
plot(x_foots(1:200))

figure()
plot(z_foots(1:200))




%try step by step
clear all
clc
load('sln_test.mat') 
q0 = [pi/6; -pi/3; 0];
dq0 = [0;0;0.];
num_steps = 1000;

% options = ...
h = 0.001; % time step
tmax = 2; % max time that we allow for a single step
tspan = 0:h:tmax;%linspace(0,tmax, tmax/h );% from 0 to tmax with time step h
y0 = [q0; dq0];
t0 = 0;

% we define the solution as a structure to simplify the post-analyses and
% animation, here we intialize it to null. 
sln.T = {};
sln.Y = {};
sln.TE = {};
sln.YE = {};

opts = odeset ( 'Events' , @event_func, 'RelTol',1e-5);

%step 1 
[T, Y, TE, YE] = ode45(@eqns, tspan, y0, opts);
% use ode45 to solve the equations of motion (eqns.m)
i = 1;
sln.T{i} = T;
sln.Y{i} = Y;
sln.TE{i} = TE;
sln.YE{i} = YE;


% Impact map
y0 = transpose(YE);
q = y0(1:3); dq = y0(4:6);
[qn,dqn] = impact(q,dq);
y0 = [qn; dqn];
t0 = T(end);
tspan = t0:h:t0+tmax;


%step 2
[T, Y, TE, YE] = ode45(@eqns, tspan, y0, opts);
% use ode45 to solve the equations of motion (eqns.m)
i = 2;
sln.T{i} = T;
sln.Y{i} = Y;
sln.TE{i} = TE;
sln.YE{i} = YE;


% Impact map
y0 = transpose(YE);
q = y0(1:3); dq = y0(4:6);
[qn,dqn] = impact(q,dq);
y0 = [qn; dqn];
t0 = T(end);
tspan = t0:h:t0+tmax;

