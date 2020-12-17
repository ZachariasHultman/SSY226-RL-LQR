function check_val(A,B,x_init, u_init, iter, K_N, n, m, t)

x = zeros(n,iter+1);
u = zeros(m,iter+1);

x(:,1) = x_init;
u(:,1) = u_init;

for i = 1:iter
   x(:,i+1) = A*x(:,i) + B*u(:,i) ;
   
   if i< 60
       noise = GenerateNoise(t(:,i));
   else
       noise = 0;
   end
   
   u(:,i+1) = -K_N*x(:,i+1) + noise;
end

figure()
plot([1:iter+1],x)
hold on
plot([1:iter+1],u)
title("Validation plot")
legend("x1", "x2", "u")
end

