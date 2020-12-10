%% 
% Evaluate the gravity matrix G given q
function G = eval_G(q)
[m1, m2, m3, l1, l2, l3, g] = set_parameters();

q1 = q(1);
q2 = q(2);
q3 = q(3);
G = [g.*l1.*sin(q1).*(m1+m2.*2.0+m3.*2.0).*(-1.0./2.0);(g.*l2.*m2.*sin(q2))./2.0;g.*l3.*m3.*sin(q3).*(-1.0./2.0)];

end