function diff_Si = diff_Si_func(U, Udot, n, m)

s = 0.5*(length(U)*(length(U)+1));
diff_Si = zeros(s,1);
c = 1;
temp = 1;

for i = 1:n+m
    for j = temp:n+m
        if i == j
           diff_Si(c) = 2*U(i)*Udot(i);
        else
            diff_Si(c) = U(i)*Udot(j) + U(j)*Udot(i);
        end
        c = c+1;        
    end
    temp = temp +1;
end

end

