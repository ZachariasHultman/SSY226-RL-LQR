
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import control as ctrl
from dynamic_system_simulation import cart_pendulum_sim_lqr
from dynamic_system_simulation import cart_pendulum_sim_lqr2
from dynamic_system_simulation import func, double_integrator_with_friction, double_integrator_with_friction2, double_integrator_with_friction_ODE, double_integrator_with_friction2_ODE
from tools import cart_pendulum_lin_lqr_gain, double_integrator_lin_lqr_gain, norm_error, mat_to_vec, mat_to_vec_sym
from AnimationFunction import animationfunction
import critic
import actor

T = 0.1  # delta t [s]
t_span =[0, 2]  # Time span for simulation
t_eval = np.linspace(t_span[0],t_span[1],int(1/T))  # Time span for simulation
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
            # [ 0,         0,0,100]]])

# R = 0.001

# K = cart_pendulum_lin_lqr_gain(L, m_p, M_p, g, f, b, M, R)  # LQR gain for linearized system
# args_lqr = (K, L, m_p, M_p, g, f, b)  # arguments for controlled linear system
# vals_lqr = integrate.solve_ivp(cart_pendulum_sim_lqr, t_span, x_init_lqr, args=args_lqr, t_eval=t_eval)
# ___________________________________________________________________________________________________________________________________________



x_init_double_int=[0, 1]  # Initial state. pos, vel, theta, thetadot for linearized system
M = np.array([[ 1,         0],
            [ 0,         1]])

R = 10
K_lqr, P = double_integrator_lin_lqr_gain(M, R)
args = (K_lqr,)
# print(K_lqr)
# vals_lqr = integrate.solve_ivp(double_integrator_with_friction, t_span, x_init_double_int, t_eval=t_eval, args=args)
vals_lqr=integrate.odeint(double_integrator_with_friction_ODE, x_init_double_int,  t_eval, args=((K_lqr,)))
# func, x_ac[:,-1], t_span_ac, args=args_ac, mxstep=1, full_output=True)


# Simulation with Actor-Critic
n = 2
m = 1
s = int(1 / 2 * ((n + m) * (n + m + 1)))
t_eval = np.linspace(t_span[0], t_span[1], int(1/T))
x_ac = np.array(x_init_double_int)
x_ac = np.atleast_2d(x_ac)
x_prev = x_ac

K = [-1, 0]

t_ac = np.ndarray(shape=(1))
error_K_ac=np.ndarray(shape=(1))
error_W_c=np.ndarray(shape=(1))

A = np.array([[0, -1],
                [0, -0.1]])

B = np.array([0, 1]).T
Q_xu=np.matmul(P,B).T
delta=1

#ekv 28
# alpha_a_upper=(1/delta*np.max(np.linalg.eigvals(np.linalg.inv(np.atleast_2d(R)))))*(2*np.min(np.linalg.eigvals(M+np.matmul(Q_xu,np.matmul(np.linalg.inv(np.atleast_2d(R)),Q_xu.T))))-np.max(np.linalg.eigvals(np.matmul(Q_xu,Q_xu.T))))
# print(alpha_a_upper)

alpha_c = 500
alpha_a = 2
s = int(1 / 2 * ((n + m) * (n + m + 1)))
#k0 = [-1, 0]
#u=np.matmul(k0, x_ac)
#u=np.atleast_1d(u)
u_prev = np.zeros(m)
u_prev=np.atleast_2d(u_prev)


t_span_ac=(0, 0)

explore=100

t_prev=0

Q_xx=P+M+np.matmul(P,A)+np.matmul(A.T,P)
Q_uu=R
W_c_opt=mat_to_vec_sym(Q_xx,n)
W_c_opt=np.concatenate((W_c_opt,mat_to_vec(Q_xu,n,m)))
W_c_opt=np.concatenate((W_c_opt,mat_to_vec_sym(Q_uu,m)))

W_c_hat = np.ones(s)
W_c_tilde = np.ones(s)
W_a_hat = np.array([-1, 0])
W_a_hat = np.atleast_2d(W_a_hat)

while t_span_ac[1]<=t_span[1]:
    # print(u_prev)
    # if t_span_ac[1]>= 1:
    # explore=explore - explore/100
    # alpha_a=alpha_a - alpha_a/100
    # alpha_c=alpha_c-alpha_c/100
    # # if alpha_a <=0:
    #   alpha_a=0
    # if alpha_c <=0:
    #   alpha_c=0
    # if explore <= 0:
    #   explore=0
 
    #args_ac = (double_integrator_with_friction2, n, m, x_prev, u_prev, alpha_c, alpha_a, M, R, T, explore,u)


    u = np.matmul(np.atleast_2d(x_ac[-1]), W_a_hat.T) + np.random.normal(0, explore, m)
    u = np.atleast_1d(u)


    W_c_hat_dot, W_c_tilde_dot, Q_xu_tilde = critic.approx_update(x_ac[-1], x_prev, u, u_prev, W_c_hat, W_c_tilde, alpha_c, M, R,
                                                                  T, n, m)
    W_a_hat_dot, W_a_tilde_dot = actor.approx_update(x_ac[:n, -1], Q_xu_tilde, W_a_hat, W_c_hat, n, m, alpha_a)

    W_c_hat = W_c_hat_old + W_c_hat_dot * T
    W_c_hat_old = W_c_hat
    W_a_hat = W_a_hat_old + W_a_hat_dot * T
    W_a_hat_old = W_a_hat

    x_prev = x_ac[:n, -1]

    x_1 = -x_ac[1, -2]*T + x_ac[0, -1]
    x_2 = (-0.1 * x_ac[1, -2] + u)*T + x_ac[1, -2]

    #args_ac = (W_a_hat,)
    t_span_ac = (t_span_ac[1], t_span_ac[1] + T)
    #vals_ac = integrate.solve_ivp(double_integrator_with_friction, t_span_ac, x_ac[:n, -1], args=args_ac, t_eval=t_span_ac)


    u_prev=u
    x_prev = x_ac[:n, -1]


    # print(u)
    #K_N = vals_ac[-1][n:n+n]
    # Save time and state values

    x_ac = np.concatenate((x_ac, [x_1, x_2]), axis=1)
    t_ac = np.concatenate((t_ac, t_span_ac[1]), axis=0)
    #e = norm_error(K_lqr, K_N)

    #e_W_c=norm_error(W_c_opt,vals_ac.y[n+n*m:n+n*m+s,-1] )
    #error_W_c=np.concatenate((error_W_c, [e_W_c]), axis=0)
    #error_K_ac = np.concatenate((error_K_ac, [e]), axis=0)
    print('time', t_span_ac[1])
    # print('states',x_ac[:n, -1])
    # print('error', e)
    #print('W_c_error', e_W_c)
    #print('W_c' , vals_ac.y[n+n*m:n+n*m+s,-1])
    #print('W_c_opt',W_c_opt)




#ani_vals = x_ac[:4]
#animationfunction(ani_vals, t_ac ,L)

plt.figure()
# Plotting controlled linear system
plt.subplot(211)
plt.plot(vals_lqr.t,vals_lqr.y[:1].T,label='x1')
plt.plot(vals_lqr.t,vals_lqr.y[1:2].T,label='x2')
plt.legend(loc="upper left")


# Plotting controlled linear system
plt.subplot(212)
plt.plot(t_ac,x_ac[:1].T,label='x1')
plt.plot(t_ac,x_ac[1:2].T,label='x2')
plt.legend(loc="upper left")
"""
# Plotting error of k
plt.subplot(413)
plt.plot(error_K_ac,label='k_error')
plt.legend(loc="upper left")

# Plotting error of Wc
plt.subplot(414)
plt.plot(error_W_c,label='W_c_error')
plt.legend(loc="upper left")
plt.show()
"""

"""
# Plotting controlled linear system
plt.plot(vals_lqr.t,vals_lqr.y[:1].T,label='pos')
plt.plot(vals_lqr.t,vals_lqr.y[1:2].T,label='vel')
plt.plot(vals_lqr.t,vals_lqr.y[2:3].T,label='theta')
plt.plot(vals_lqr.t,vals_lqr.y[3:4].T,label='theta_dot')
plt.legend(loc="upper left")
plt.show()
"""

