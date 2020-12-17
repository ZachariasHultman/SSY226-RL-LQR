function D = GetVec2mat(v,n , m)

c = 1;

D = zeros(n+m);

for i = 1:n+m
    for j = 1:n+m
        if i<= j
            D(j,i) = v(c);
            c = c+1;
        end
    end
end

end

