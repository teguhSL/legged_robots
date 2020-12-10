function results = analyse(sln, parameters, to_plot)
% calculate gait quality metrics (distance, step frequency, step length,
% velocity, etc.)
x_hs = [];
dx_hs_mean = [];
dx_hs = [];

z_hs = [];
dz_hs = [];

qs = [];
dqs = [];
us = [];

t_hs = [];

fs = [];
ds = [];
effort = 0;
cot = 0;
for i=1:size(sln.Y,2)
   Y = sln.Y{i};
   T = sln.T{i};
   t_hs = [t_hs; T];
   dt = T(end) - T(1);
   f = 1.0/dt;
   fs = [fs f];
   
   for j=1:size(Y,1)
       y = Y(j,:);
       
       [x_h, z_h, dx_h, dz_h] =  kin_hip(y(1:3), y(4:6));
       x_hs = [x_hs; x_h];
       dx_hs = [dx_hs; dx_h];
       z_hs = [z_hs; z_h];
       dz_hs = [dz_hs; dz_h];
       
       qs = [qs; y(1:3)];
       dqs = [dqs; y(4:6)];
       u = control(0, y(1:3), y(4:6),0,0,0,parameters);
       us = [us u];
   end
   y = Y(1,:);
   [x_h0, z_h0, dx_h, dz_h] =  kin_hip(y(1:3), y(4:6));
   y = Y(end,:);
   [x_hT, z_hT, dx_h, dz_h] =  kin_hip(y(1:3), y(4:6));
   d = x_hT - x_h0;
   ds = [ds d];
   
   dx_hs_mean = [dx_hs_mean; d/dt];
    
end

results = struct();
results.x_h = x_hs;
results.dx_h = dx_hs;
results.z_h = z_hs;
results.dz_h = dz_hs;
results.t_h = t_hs;

results.qs = qs;
results.dqs = dqs;
results.us = us;

results.fs = fs;
results.ds = ds;

T = results.t_h(end);
Umax = 30;
results.effort = sum(results.us.^2, 'all')/(2*T*Umax);
results.cot = results.effort/sum(results.ds);

results.dx_hs_mean = dx_hs_mean;
results.mean_velocity = sum(results.ds)/T;

% calculate actuation (you can use the control function)

if to_plot
    % plot the angles
    figure();
    plot(results.qs(:,1), results.dqs(:,1));
    xlabel('q1');
    ylabel('dq1');
    
    figure();
    plot(results.qs(:,2), results.dqs(:,2));
    xlabel('q2');
    ylabel('dq2');

    figure();
    plot(results.qs(:,3), results.dqs(:,3));
    xlabel('q3');
    ylabel('dq3');

    % plot the hip position
    figure();
    plot(results.t_h, results.x_h);
    xlabel('t');
    ylabel('xh');
    
    figure();
    plot(results.t_h, results.z_h);
    xlabel('t');
    ylabel('zh');

    % plot instantaneous and average velocity
    figure();
    plot(results.t_h, results.dx_h);
    xlabel('t');
    ylabel('dxh');
    
   
    figure();
    plot(results.dx_hs_mean);
    xlabel('step number');
    ylabel('dxh_mean');

    % plot frequency and displacement
    step_length = size(results.fs,2);
    figure();
    plot(2:1:step_length, results.fs(2:end));
    xlabel('step number');
    ylabel('freq');

    figure();
    plot(2:1:step_length, results.ds(2:end));
    xlabel('step number');
    ylabel('disp');
    
    
    % plot actuation
    figure();
    plot(results.t_h, results.us(1,:));
    xlabel('t');
    ylabel('u1');
    
    
    figure();
    plot(results.t_h, results.us(2,:));
    xlabel('t');
    ylabel('u2');

end

end