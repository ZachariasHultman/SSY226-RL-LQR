import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import control as ctrl
from dynamic_system_simulation import cart_pendulum_sim
from dynamic_system_simulation import cart_pendulum_sim_lqr
from tools import cart_pendulum_lin_lqr_gain
from actor import actor
from critic import critic

M = 0.5  # cart mass
m = 0.2  # pendulum mass
g = 9.81  # gravity
L = 0.3  # pendulum length
f = 0.1  # friction
b= 0.5 #friction for cart
F = 1  # control input [N]

x_init = [0, 0, np.pi, 0]  # Initial state. pos, vel, theta, thetadot
x_init_lqr=[0, 0, 0.25*np.pi, 0]  # Initial state. pos, vel, theta, thetadot for linearized system

t_span =[0, 10]  # Time span for simulation
t_eval = np.linspace(t_span[0],t_span[1],500)  # Time span for simulation

Q = np.array([[ 1,         0,                 0,                  0],
            [ 0,         1,                 0,                  0],
            [ 0,         0,                 10,                  0],
            [ 0,         0,                 0,                  100]])

R = 0.01

K = cart_pendulum_lin_lqr_gain(L, m, M, g, f, b, Q, R)  # LQR gain for linearized system

args = (L, m, M, g, F, f, b)  # arguments for non-linear system
args_lqr = (K, L, m, M, g, f, b)  # arguments for controlled linear system

vals = integrate.solve_ivp(cart_pendulum_sim, t_span, x_init, args=args, t_eval=t_eval)
vals_lqr = integrate.solve_ivp(cart_pendulum_sim_lqr, t_span, x_init_lqr, args=args_lqr, t_eval=t_eval)
# x_vals = integrate.odeint(cart_pendulum_sim, x_init, t_span)

# Simulation with Actor-Critic
T = 0.05  # delta t [s]
t_eval = np.linspace(t_span[0], t_span[1], 200)
x_ac = np.ndarray(shape=(4, 1))
x_ac[:, 0] = x_init_lqr
t_ac = np.ndarray(shape=(1, 0))
actor = actor(n=4, m=1, alpha=2)
critic = critic(n=4, m=1, alpha=50)
u_prev = np.zeros(1)
x_prev = x_ac[:, 0]
for t in t_eval:
    u = -np.matmul(K, x_ac[:, -1])
    u = np.atleast_1d(u)

    x = x_ac[:, -1]
    critic.approx_update(x, x_prev, u, u_prev, Q, R, T)
    actor.approx_update(critic.Q_uu(), critic.Q_ux(), x)
    K = actor.W
    u_prev = u
    x_prev = x
    args_ac = (K, L, m, M, g, f, b)

    t_span_ac = [t, t+T]

    vals_ac = integrate.solve_ivp(cart_pendulum_sim_lqr, t_span_ac, x_ac[:, -1], args=args_ac)

    # Save time and state values
    x_ac = np.concatenate((x_ac, vals_ac.y), axis=1)
    t_ac = np.concatenate((t_ac, vals_ac.t), axis=0)




# Plotting non-linear system
plt.plot(vals.t,vals.y[:1].T,label='pos')
plt.plot(vals.t,vals.y[1:2].T,label='vel')
plt.plot(vals.t,vals.y[2:3].T,label='theta')
plt.plot(vals.t,vals.y[3:4].T,label='theta_dot')
plt.legend(loc="upper left")
plt.show()

# Plotting controlled linear system
plt.plot(vals_lqr.t,vals_lqr.y[:1].T,label='pos')
plt.plot(vals_lqr.t,vals_lqr.y[1:2].T,label='vel')
plt.plot(vals_lqr.t,vals_lqr.y[2:3].T,label='theta')
plt.plot(vals_lqr.t,vals_lqr.y[3:4].T,label='theta_dot')
plt.legend(loc="upper left")
plt.show()

# Plotting controlled linear system
plt.plot(t_ac,x_ac[:1].T,label='pos')
plt.plot(t_ac,x_ac[1:2].T,label='vel')
plt.plot(t_ac,x_ac[2:3].T,label='theta')
plt.plot(t_ac,x_ac[3:4].T,label='theta_dot')
plt.legend(loc="upper left")
plt.show()