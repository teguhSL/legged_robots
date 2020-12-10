%% 
% computing the energy
% 
function [q_p, dq_p] = impact(q_m, dq_m)
q_p(1, 1) = q_m(2, 1); % angular positions before/after impact
q_p(2, 1) = q_m(1, 1) ;
q_p(3, 1) = q_m(3, 1) ;

A_m = eval_A_m(q_m);
A_p = eval_A_p(q_p);

% A_p dq_p = A_m dq_m
% Note: To solve the equation Ax = b you could use x = A \ b to avoid taking
% inverse of A. 
dq_p = A_p \ A_m*dq_m;
end