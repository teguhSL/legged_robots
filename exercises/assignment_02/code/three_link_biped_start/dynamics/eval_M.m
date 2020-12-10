%%
% Evaluate the Mass matrix given q
function M = eval_M(q)
[m1, m2, m3, l1, l2, l3, g] = set_parameters();

q1 = q(1);
q2 = q(2);
q3 = q(3);

t2 = -q2;
t3 = -q3;
t4 = q1+t2;
t5 = q1+t3;
t6 = cos(t4);
t7 = cos(t5);
t8 = (l1.*l2.*m2.*t6)./2.0;
t9 = (l1.*l3.*m3.*t7)./2.0;
t10 = -t8;
M = reshape([l1.^2.*(m1./4.0+m2+m3),t10,t9,t10,(l2.^2.*m2)./4.0,0.0,t9,0.0,(l3.^2.*m3)./4.0],[3,3]);

end