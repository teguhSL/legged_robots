function [T, V] = eval_energy(q, dq)
[m1, m2, m3, l1, l2, l3, g] = set_parameters();
q1 = q(1);
q2 = q(2);
q3 = q(3);
dq1 = dq(1);
dq2 = dq(2);
dq3 = dq(3);

x1 = (l1*sin(q1))/2;
z1 = (l1*cos(q1))/2;
x2 = l1*sin(q1) - (l2*sin(q2))/2;
z2 = l1*cos(q1) - (l2*cos(q2))/2;
x3 = l1*sin(q1) + (l3*sin(q3))/2;
z3 = l1*cos(q1) + (l3*cos(q3))/2;
dx1 = (dq1*l1*cos(q1))/2;
dz1 = -(dq1*l1*sin(q1))/2;
dx2 = dq1*l1*cos(q1) - (dq2*l2*cos(q2))/2;
dz2 = (dq2*l2*sin(q2))/2 - dq1*l1*sin(q1);
dx3 = dq1*l1*cos(q1) + (dq3*l3*cos(q3))/2;
dz3 = - dq1*l1*sin(q1) - (dq3*l3*sin(q3))/2;


T1 = 0.5*m1*(dx1^2 + dz1^2);
T2 = 0.5*m2*(dx2^2 + dz2^2);
T3 = 0.5*m3*(dx3^2 + dz3^2);
T = T1 + T2 + T3;

% V1, V2, V3: potential energies of m1, m2, m3
V1 = m1*g*z1;
V2 = m2*g*z2;
V3 = m3*g*z3;
V = V1 + V2 + V3;
end