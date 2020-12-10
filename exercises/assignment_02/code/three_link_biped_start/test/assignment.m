addpath('../dynamics/', '../set_parameters/');
%question 1
q_m = [0.6324 ;   -0.6324  ;  0.0975]; %q1 and q2 have to be the same
dq_m = [0.2785 ;   0.5469  ;  0.9575];

[q_p, dq_p] = impact(q_m, dq_m);
[T_m, V_m] = eval_energy(q_m, dq_m);
[T_p, V_p] = eval_energy(q_p, dq_p);

fprintf('T_p - T_m: \n')
disp(T_p-T_m)

fprintf('V_p - V_m: \n')
disp(V_p-V_m)

%question 2
q_m = [pi/6; -pi/6; pi/10];
dq_m = [1; 0.2; 0]; 
[q_p, dq_p] = impact(q_m, dq_m);
[T_m, V_m] = eval_energy(q_m, dq_m);
[T_p, V_p] = eval_energy(q_p, dq_p);
dT = 100*(T_m-T_p)/T_m;
fprintf('The decrease in kinetic energy is :')
disp(dT)


%question 3
i=1;
for alpha=0:0.02:pi/4
q_m = [alpha; -alpha; 0];
dq_m = [1; 0.2; 0]; 
[q_p, dq_p] = impact(q_m, dq_m);
[T_m, V_m] = eval_energy(q_m, dq_m);
[T_p, V_p] = eval_energy(q_p, dq_p);
dT = 100*(T_m-T_p)/T_m;
dTs(i) = dT;
alphas(i) = alpha;
i = i + 1;
end

plot(alphas, dTs)
title('Effect of the angle alpha to the Kinetic energy loss')
xlabel('alpha (rad)')
ylabel('Kinetic enery loss (%)')
