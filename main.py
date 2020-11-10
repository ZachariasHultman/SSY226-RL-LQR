import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from dynamic_system_simulation import cart_pendulum_sim



M = 0.5  # cart mass
m = 0.2  # pendulum mass
g = 9.81  # gravity
L = 0.3  # pendulum length
f = 0  # friction
f_cart= 1 #friction for cart
F = 1  # control input [N]


x_init = [0, 0, np.pi, 0]  # Initial state. pos, vel, theta, thetadot

t_span =[0, 10]  # Time span for simulation
t_eval =np.linspace(t_span[0],t_span[1],50)  # Time span for simulation


args = (L, m, M, g, F, f, f_cart)

vals = integrate.solve_ivp(cart_pendulum_sim, t_span, x_init, args=args, t_eval=t_eval)
# x_vals = integrate.odeint(cart_pendulum_sim, x_init, t_span)
print(vals.y.shape)
# print(vals.y[:2])

plt.plot(vals.t,vals.y[:1].T,label='pos')
plt.plot(vals.t,vals.y[1:2].T,label='vel')
plt.plot(vals.t,vals.y[2:3].T,label='theta')
plt.plot(vals.t,vals.y[3:4].T,label='theta_dot')
plt.legend(loc="upper left")
plt.show()