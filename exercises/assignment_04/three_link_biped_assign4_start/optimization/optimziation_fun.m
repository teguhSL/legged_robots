function objective_value = optimziation_fun(parameters)
% extract parameters q0, dq0 and x
q0 = parameters(1:3);
dq0 = parameters(4:6);
x = parameters(7:end);

% run simulation
num_steps = 10; % the higher the better, but slow
sln = solve_eqns(q0, dq0, num_steps, x);
results = analyse(sln, x, false);

% calculate metrics such as distance, mean velocity and cost of transport
max_actuation = 30;
effort = results.effort;
distance = sum(results.ds);
velocity = results.mean_velocity;
CoT = results.cot;
w1 = 1;
w2 = 0.001;
vd = .9;
objective_value = w1*abs(vd - velocity) + w2*CoT;

% handle corner case when model walks backwards (e.g., objective_value =
% 1000)

if distance < 0
    objective_value = 1000;
end

% handle case when model falls (e.g., objective_value = 1000)
disp(objective_value)
end

