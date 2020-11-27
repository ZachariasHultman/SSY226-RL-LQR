
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import control as ctrl
from dynamic_system_simulation import cart_pendulum_sim_lqr
from dynamic_system_simulation import cart_pendulum_sim_lqr2
from dynamic_system_simulation import func, double_integrator_with_friction, double_integrator_with_friction2
from tools import cart_pendulum_lin_lqr_gain, double_integrator_lin_lqr_gain, norm_error
from AnimationFunction import animationfunction

t_span =[0, 4]  # Time span for simulation
t_eval = np.linspace(t_span[0],t_span[1],500)  # Time span for simulation

# M_p = 0.5  # cart mass
# m_p = 0.2  # pendulum mass
# g = 9.81  # gravity
# L = 2  # pendulum length
# f = 0.1  # friction
# b= 0.5 #friction for cart
# F = 1  # control input [N]

# x_init = [0, 0, 0.05*np.pi, 0]  # Initial state. pos, vel, theta, thetadot
# x_init_lqr=[0, 0, 0.05*np.pi, 0]  # Initial state. pos, vel, theta, thetadot for linearized system~

# M = np.array([[ 1,         0, 0,0],
#             [ 0,         1,0,0],
#             [ 0,         0,100,0,
#             [ 0,         0,0,100]]])

# R = 0.001

# K = cart_pendulum_lin_lqr_gain(L, m_p, M_p, g, f, b, M, R)  # LQR gain for linearized system
# args_lqr = (K, L, m_p, M_p, g, f, b)  # arguments for controlled linear system
# vals_lqr = integrate.solve_ivp(cart_pendulum_sim_lqr, t_span, x_init_lqr, args=args_lqr, t_eval=t_eval)
# ___________________________________________________________________________________________________________________________________________



x_init_double_int=[0, 1]  # Initial state. pos, vel, theta, thetadot for linearized system
M = np.array([[ 1,         0],
            [ 0,         1]])

R = 10
K_lqr = double_integrator_lin_lqr_gain(M, R)
args = (K_lqr,)
print(K_lqr)
vals_lqr = integrate.solve_ivp(double_integrator_with_friction, t_span, x_init_double_int, t_eval=t_eval, args=args)



# Simulation with Actor-Critic
n = 2
m = 1
s = int(1 / 2 * ((n + m) * (n + m + 1)))
T = 0.05  # delta t [s]
t_eval = np.linspace(t_span[0], t_span[1], int(1/T))
x_ac = np.ones(shape=(n+n+s+s, 1))
x_ac[:n, 0] = x_init_double_int
K =[-1, 0]
x_ac[n:n+n*m,0]=K

t_ac = np.ndarray(shape=(1))
error_K_ac=np.ndarray(shape=(1))

Q_xu=np.atleast_2d(R)
delta=1
alpha_a_upper=(1/delta*np.max(np.linalg.eigvals(np.linalg.inv(np.atleast_2d(R)))))*(2*np.min(np.linalg.eigvals(M+np.matmul(Q_xu,np.matmul(np.linalg.inv(np.atleast_2d(R)),Q_xu.T))))-np.max(np.linalg.eig(np.matmul(Q_xu,Q_xu.T))))
print(alpha_a_upper)

alpha_c = 400
alpha_a = 5
s = int(1 / 2 * ((n + m) * (n + m + 1)))

# M = np.identity(n)
# R = np.identity(m)

u=np.matmul(np.ones((n,m)).T,x_ac[:n, 0])
u=np.atleast_1d(u)
u_prev=0
u_prev=np.atleast_1d(u_prev)

x_prev=x_ac[:n, 0]
flag=True
errorFlag=False
t_span_ac=(0, 0)

explore=0.1


t_prev=0

while t_span_ac[1]<=t_span[1]:
  # print(u_prev)
  if t_span_ac[1]>= 1:
    explore=explore - explore/100
    # alpha_a=alpha_a - alpha_a/100
    # alpha_c=alpha_c-alpha_c/100
    # # if alpha_a <=0:
    #   alpha_a=0
    # if alpha_c <=0:
    #   alpha_c=0
    # if explore <= 0:
    #   explore=0
 
  args_ac = (double_integrator_with_friction2, n, m, x_prev, u_prev, alpha_c, alpha_a, M, R, T, explore)

  t_span_ac = (t_span_ac[1], t_span_ac[1]+T)

  #try:
  vals_ac = integrate.solve_ivp(func, t_span_ac, x_ac[:,-1], args=args_ac)
  #except :
    #   print('RuntimeError is raised!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    #  errorFlag=True

  u_prev=u
  x_prev= x_ac[:n, -1]
  u = np.matmul(vals_ac.y[n:n+n, -1], vals_ac.y[:n, -1]) + np.random.normal(0,np.abs(u)*explore,m)
  u=np.atleast_1d(u)
  # print(u)
  K_N = vals_ac.y[n:n+n, -1]
  # Save time and state values
  print('K_N',K_N)
  x_ac = np.concatenate((x_ac, vals_ac.y), axis=1)
  t_ac = np.concatenate((t_ac, vals_ac.t), axis=0)
  e = norm_error(K_lqr, K_N)
 
  error_K_ac = np.concatenate((error_K_ac, [e]), axis=0)
  print('time',t_span_ac[1])
  print('states',x_ac[:n, -1])
  print('error', e)


#ani_vals = x_ac[:4]
#animationfunction(ani_vals, t_ac ,L)

plt.figure()
# Plotting controlled linear system
plt.subplot(311)
plt.plot(vals_lqr.t,vals_lqr.y[:1].T,label='x1')
plt.plot(vals_lqr.t,vals_lqr.y[1:2].T,label='x2')
plt.legend(loc="upper left")


# Plotting controlled linear system
plt.subplot(312)
plt.plot(t_ac,x_ac[:1].T,label='x1')
plt.plot(t_ac,x_ac[1:2].T,label='x2')
plt.legend(loc="upper left")

# Plotting error of k
plt.subplot(313)
plt.plot(error_K_ac,label='k_error')
plt.legend(loc="upper left")
plt.show()


"""
# Plotting controlled linear system
plt.plot(vals_lqr.t,vals_lqr.y[:1].T,label='pos')
plt.plot(vals_lqr.t,vals_lqr.y[1:2].T,label='vel')
plt.plot(vals_lqr.t,vals_lqr.y[2:3].T,label='theta')
plt.plot(vals_lqr.t,vals_lqr.y[3:4].T,label='theta_dot')
plt.legend(loc="upper left")
plt.show()
"""

