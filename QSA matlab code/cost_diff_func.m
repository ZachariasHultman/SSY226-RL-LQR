function diff_d = cost_diff_func(x, xdot, u, udot, M, R)

diff_d = 2*x'*M*xdot + 2*u'*R*udot;

end

