function A_m = eval_A_m(q_m)
[m1, m2, m3, l1, l2, l3, g] = set_parameters();
m=m1;
q1_m = q_m(1);
q2_m = q_m(2);
q3_m = q_m(3);
t2 = -q2_m;
t3 = -q3_m;
t4 = q1_m+t2;
t5 = q1_m+t3;
t6 = cos(t4);
t7 = cos(t5);
A_m = reshape([l1.*(-l1.*m+l2.*m.*t6.*4.0+l2.*m3.*t6.*4.0+l3.*m3.*t7.*2.0).*(-1.0./4.0),(l1.^2.*m)./4.0,l1.*l3.*m3.*t7.*(-1.0./2.0),(l2.^2.*m)./4.0,0.0,0.0,l3.*m3.*(l3+l2.*cos(q2_m+t3).*2.0).*(-1.0./4.0),0.0,l3.^2.*m3.*(-1.0./4.0)],[3,3]);

end