function cost = cost_func(x, u, M, R)

cost = x'*M*x + u'*R*u;

end

