function kron_vec = GetKron(U, n, m)

temp = kron(U,U);

mat = vec2mat(temp, n+m);

uTriangular = tril(true(size(mat)),0);

kron_vec = temp(uTriangular);

end

