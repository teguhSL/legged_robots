% run the following code
clear all
load('sln_test.mat') 
q0 = [pi/6; -pi/3; 0];
dq0 = [0;0;0.];
num_steps = 10;
sln = solve_eqns(q0, dq0, num_steps);
animate(sln);

error = 0 ;
for i = 1:num_steps
    error = error + norm(sln.Y{i} - sln_test.Y{i});
end

if(error < 1e-5)
    disp('Great! your answer is right!')
else
    disp('OPS! Something is wrong! Please check your code carefully!')
end