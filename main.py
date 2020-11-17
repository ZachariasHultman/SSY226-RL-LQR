import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import control as ctrl
from dynamic_system_simulation import cart_pendulum_sim
from dynamic_system_simulation import cart_pendulum_sim_lqr
from dynamic_system_simulation import cart_pendulum_sim_lqr2
from dynamic_system_simulation import func
from tools import cart_pendulum_lin_lqr_gain


M_p = 0.5  # cart mass
m_p = 0.2  # pendulum mass
g = 9.81  # gravity
L = 1  # pendulum length
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

K = cart_pendulum_lin_lqr_gain(L, m_p, M_p, g, f, b, Q, R)  # LQR gain for linearized system

args = (L, m_p, M_p, g, F, f, b)  # arguments for non-linear system
args_lqr = (K, L, m_p, M_p, g, f, b)  # arguments for controlled linear system

vals = integrate.solve_ivp(cart_pendulum_sim, t_span, x_init, args=args, t_eval=t_eval)
vals_lqr = integrate.solve_ivp(cart_pendulum_sim_lqr, t_span, x_init_lqr, args=args_lqr, t_eval=t_eval)
# x_vals = integrate.odeint(cart_pendulum_sim, x_init, t_span)

# Simulation with Actor-Critic
n = 4
m = 1
s = int(1 / 2 * ((n + m) * (n + m + 1)))
T = 0.05  # delta t [s]
t_eval = np.linspace(t_span[0], t_span[1], 200)
x_ac = np.zeros(shape=(n+n+s+s, 1))
x_ac[:n, 0] = x_init_lqr

t_ac = np.ndarray(shape=(1, 0))

alpha_c = 50
alpha_a = 2
s = int(1 / 2 * ((n + m) * (n + m + 1)))
K = np.ones((n,m))
M = np.identity(n)
R = np.identity(m)
for t in t_eval:

    u = -np.matmul(K.T, x_ac[:n, -1])
    
    u = np.atleast_1d(u)

    x = x_ac[:n, -1]

    u_prev = u
    x_prev = x
    args_ac = (cart_pendulum_sim_lqr2, L, m_p, M_p, g, F, f, b, n , m, x_prev, u_prev, alpha_c, alpha_a, M, R, T)

    t_span_ac = [t, t+T]

    vals_ac = integrate.solve_ivp(func, t_span_ac, x_ac[:, -1], args=args_ac)

    K = vals_ac.y[n:s]
    # Save time and state values
    x_ac = np.concatenate((x_ac, vals_ac.y[:n]), axis=1)
    t_ac = np.concatenate((t_ac, vals_ac.t), axis=0)

    print("eureka")


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
