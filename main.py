import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import control as ctrl
from dynamic_system_simulation import cart_pendulum_sim
from dynamic_system_simulation import cart_pendulum_sim_lqr
from tools import cart_pendulum_lin_lqr_gain
from AnimationFunction import animationfunction


M = 0.5  # cart mass
m = 0.2  # pendulum mass
g = 9.81  # gravity
L = 1  # pendulum length
f = 0.1  # friction
b= 0.5 #friction for cart
F = 1  # control input [N]

x_init = [0, 0, np.pi, 0]  # Initial state. pos, vel, theta, thetadot
x_init_lqr=[0, 0, 0.25*np.pi, 0]  # Initial state. pos, vel, theta, thetadot for linearized system

t_span =[0, 10]  # Time span for simulation
t_eval =np.linspace(t_span[0],t_span[1],500)  # Time span for simulation

K = cart_pendulum_lin_lqr_gain(L, m, M, g, f, b) # LQR gain for linerized system

args = (L, m, M, g, F, f, b) #arguments for non-linear system
args_lqr = (K,L, m, M, g, f, b) #arguments for controlled linear system

vals = integrate.solve_ivp(cart_pendulum_sim, t_span, x_init, args=args, t_eval=t_eval)
vals_lqr = integrate.solve_ivp(cart_pendulum_sim_lqr, t_span, x_init_lqr, args=args_lqr, t_eval=t_eval)
# x_vals = integrate.odeint(cart_pendulum_sim, x_init, t_span)

animationfunction(vals_lqr, L)


# # Plotting non-linear system
# plt.plot(vals.t,vals.y[:1].T,label='pos')
# plt.plot(vals.t,vals.y[1:2].T,label='vel')
# plt.plot(vals.t,vals.y[2:3].T,label='theta')
# plt.plot(vals.t,vals.y[3:4].T,label='theta_dot')
# plt.legend(loc="upper left")
# plt.show()

# # Plotting controlled linear system
# plt.plot(vals_lqr.t,vals_lqr.y[:1].T,label='pos')
# plt.plot(vals_lqr.t,vals_lqr.y[1:2].T,label='vel')
# plt.plot(vals_lqr.t,vals_lqr.y[2:3].T,label='theta')
# plt.plot(vals_lqr.t,vals_lqr.y[3:4].T,label='theta_dot')
# plt.legend(loc="upper left")
# plt.show()