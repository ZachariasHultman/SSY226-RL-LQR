import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import control as ctrl
from dynamic_system_simulation import cart_pendulum_sim_lqr
from dynamic_system_simulation import cart_pendulum_sim_lqr2
from dynamic_system_simulation import func, double_integrator_with_friction, double_integrator_with_friction2, double_integrator_with_friction_ODE, double_integrator_with_friction2_ODE, test_sys_ODE, test_sys2_ODE
from tools import cart_pendulum_lin_lqr_gain, double_integrator_lin_lqr_gain, norm_error, mat_to_vec_sym, vech_to_mat_sym
from AnimationFunction import animationfunction
import critic
import actor


T = 0.05  
dt=0.001 # delta t [s]
t_span =[0, 10]  # Time span for simulation
t_eval = np.linspace(t_span[0],t_span[1],int(1/dt))  # Time span for simulation

n = 3
m = 2

A =np.array([[-1.01887, -0.90506, -0.00215],
     [0.82225, -1.07741, -0.17555],
     [0, 0, -1]])

B = np.array([[0, 0, 1], [1, 1, 1]]).T

M = 2*np.eye(n)
R = 1*np.eye(m)
# A = np.array([[0, -1],
#                 [0, -0.1]])
# B = np.array([[0],[1]])
# M = np.array([[ 1,         0],
#             [ 0,         1]])
# R = np.array([0.1])
# R= R.reshape(m,m)


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
#             [ 0,         0,100,0
            # [ 0,         0,0,100]]])

# R = 0.001

# K = cart_pendulum_lin_lqr_gain(L, m_p, M_p, g, f, b, M, R)  # LQR gain for linearized system
# args_lqr = (K, L, m_p, M_p, g, f, b)  # arguments for controlled linear system
# vals_lqr = integrate.solve_ivp(cart_pendulum_sim_lqr, t_span, x_init_lqr, args=args_lqr, t_eval=t_eval)
# ___________________________________________________________________________________________________________________________________________
yay =np.asarray([0.1000,    0.1000,   -0.1000,    0.9821,    0.5783,    0.2344,    0.8106,    0.4513,    0.2500,    0.9554,    0.1427,    0.5126,    0.9719,    0.6483,    0.6147,    0.4697,    0.5778,    0.9113,    0.3762, 0.2736, 0.2288,  0.4446,   0.4235,   0.6275,0])
W_c_hat = yay[3:18]
W_a_hat = yay[18:24].reshape(n,m)
print(W_c_hat)
print(W_a_hat)


x_init=np.array([0.1, 0.1, -0.1])

# x_init_double_int=[0, 1]  # Initial state. pos, vel, theta, thetadot for linearized system

# K_lqr, P = double_integrator_lin_lqr_gain(M, R)
K_lqr, P, E =ctrl.lqr(A, B, M, R)
# args = (K_lqr,)
# print(K_lqr)
# vals_lqr = integrate.solve_ivp(double_integrator_with_friction, t_span, x_init_double_int, t_eval=t_eval, args=args)
vals_lqr=integrate.odeint(test_sys_ODE, x_init,  t_eval, args=((K_lqr,)))
# func, x_ac[:,-1], t_span_ac, args=args_ac, mxstep=1, full_output=True)

# Simulation with Actor-Critic
s = int(1 / 2 * ((n + m) * (n + m + 1)))
x_ac = np.array(x_init)
x_ac = np.atleast_2d(x_ac).T
x_curr=x_ac

# K = np.array([[-1],[0]])
K=K_lqr

t_ac =[]
t_ac.append(0)
error_K_ac=np.ndarray(shape=(1))
error_W_c=np.ndarray(shape=(1))

u_prev = np.zeros(m)
# u_prev=np.atleast_2d(u_prev)
u_hist=u_prev.reshape(m,1)
t_span_ac=(0, 0)

Q_xx=P+M+ P@A + A.T@P
Q_uu=R
Q_xu=P @ B
Q_ux= Q_xu.T

Q_opt_upper=np.concatenate((Q_xx,Q_xu),1)
Q_opt_lower=np.concatenate((Q_ux,Q_uu),1)
Q_opt=np.concatenate((Q_opt_upper,Q_opt_lower),0)

W_c_opt=np.triu(Q_opt)
row,col=np.tril_indices(n+m)
W_c_opt=[Q_opt[c,r] for r,c in zip(col,row)]
W_c_opt=np.asarray(W_c_opt)
print(W_c_opt)
# W_c_opt=mat_to_vec_sym(Q_opt,n,m)

#ekv 28
delta=1
alpha_a_upper=(1/delta*np.max(np.linalg.eigvals(np.linalg.inv(np.atleast_2d(R)))))*((2*np.min(np.linalg.eigvals(M+(Q_xu@ np.linalg.inv(np.atleast_2d(R))@ Q_xu.T))))-np.max(np.linalg.eigvals(Q_xu@Q_xu.T)))
print(alpha_a_upper)

# W_c_hat=W_c_opt  + 1
W_c_hat=np.atleast_2d(W_c_hat).T
W_c_hat_old=W_c_hat

# W_a_hat = np.array(-K) +1
W_a_hat = np.atleast_2d(W_a_hat)
W_a_hat_old= W_a_hat
k=0

alpha_c = 50
alpha_a = 2
explore=1
k_max=int(T/dt)

int_term_prev=0
int_term=0

while t_span_ac[1]<=t_span[1]:
    if t_span_ac[1] >= 4*int(t_span[1]/5):
        explore=0
        
    # if t_span_ac[1] % t_span[1]/5 == 0:
    #     explore=2
        # alpha_c =alpha_c-alpha_c/100
    k=k+1
    # Controll signals
    u = np.matmul(W_a_hat.T,x_curr)
    u_hist=np.concatenate((u_hist, u), axis=1)
    # u_sys=u+ np.random.normal(0, explore, 1,)
    
    u_sys = u + explore*0.1*np.exp(-0.0001*t_span_ac[1])*1*(np.sin(t_span_ac[1])**2*np.cos(t_span_ac[1])+np.sin(2*t_span_ac[1])**2*np.cos(0.1*t_span_ac[1])+np.sin(-1.2*t_span_ac[1])**2*np.cos(0.5*t_span_ac[1])+np.sin(t_span_ac[1])**5+np.sin(1.12*t_span_ac[1])**2+np.cos(2.4*t_span_ac[1])*np.sin(2.4*t_span_ac[1])**3)
    # print(u_sys)
    # u_sys=u
    # print(0.1*np.exp(-0.0001*t_span_ac[1])*1*(np.sin(t_span_ac[1])**2*np.cos(t_span_ac[1])+np.sin(2*t_span_ac[1])**2*np.cos(0.1*t_span_ac[1])+np.sin(-1.2*t_span_ac[1])**2*np.cos(0.5*t_span_ac[1])+np.sin(t_span_ac[1])**5+np.sin(1.12*t_span_ac[1])**2+np.cos(2.4*t_span_ac[1])*np.sin(2.4*t_span_ac[1])**3))
    
    if k >= k_max:
        if k==k_max:
            time_step=T
        else:
            time_step=dt
        # Actor Critic learning
        W_c_hat_dot = critic.approx_update(x_ac[:,-k_max:],u_hist[:,-k_max:], W_c_hat, alpha_c, M, R, dt, n, m,int_term)
        # print(W_c_hat_dot)
        
        W_a_hat_dot = actor.approx_update(x_ac[:,-1:], W_a_hat, W_c_hat, n, m, alpha_a)   
        
        # print('dot',W_c_hat_dot)
        W_c_hat = W_c_hat_old + W_c_hat_dot * time_step
        # print('W_c_hat',W_c_hat)

        W_c_hat_old = W_c_hat
        W_a_hat = W_a_hat_old + W_a_hat_dot * time_step
        W_a_hat_old = W_a_hat
     
        e = norm_error(-K_lqr, W_a_hat.T)
        # print(e)
        e_W_c=norm_error(W_c_opt,W_c_hat.T)
        # print('e_W_c',e_W_c)
        error_W_c=np.concatenate((error_W_c, [e_W_c]), axis=0)
        error_K_ac = np.concatenate((error_K_ac, [e]), axis=0)

    t_span_ac = (t_span_ac[1], t_span_ac[1] + dt)
    # System to be simulated
    x_prev=x_ac[:,-1:]

   
    # x_1 = -x_prev[1]*dt + x_prev[0]
    # x_2 = (-0.1 * x_prev[1] + u_sys)*dt + x_prev[1]
    # print(B)
    # print(u_sys)

    x_curr = (A @ x_prev + B @ u_sys)*dt + x_prev

    int_term=int_term_prev + (x_curr.T @ M @x_curr + u.T @ R @ u) *dt
    int_term_prev=int_term
    # x_curr = np.atleast_2d([x_1 , x_2])

    x_ac = np.concatenate((x_ac, x_curr), axis=1)
    t_ac.append(t_span_ac[1])

    # print('time', t_span_ac[1])
    # print('states',x_ac[:n, -1])
    # print('error', e)
    # print('W_c_error', e_W_c)
    # print('W_c' ,W_c_hat)
    # print('W_c_opt',W_c_opt)
    # print(W_a_hat)
    # br

#animationfunction(ani_vals, t_ac ,L)

plt.figure()
# Plotting controlled linear system
plt.subplot(511)
plt.plot(t_eval,vals_lqr[:,0].T,label='x1')
plt.plot(t_eval,vals_lqr[:,1].T,label='x2')
plt.plot(t_eval,vals_lqr[:,2].T,label='x3')
plt.legend(loc="upper left")


# Plotting controlled linear system
plt.subplot(512)
plt.plot(t_ac,x_ac[0,:].T,label='x1')
plt.plot(t_ac,x_ac[1,:].T,label='x2')
plt.plot(t_ac,x_ac[2,:].T,label='x3')
plt.legend(loc="upper left")


# Plotting error of k
plt.subplot(513)
plt.plot(error_K_ac,label='k_error')
plt.legend(loc="upper left")

# Plotting error of Wc
plt.subplot(514)
plt.plot(error_W_c,label='W_c_error')
plt.legend(loc="upper left")

vals_lqr_new=integrate.odeint(test_sys_ODE, x_init,  t_eval, args=((-W_a_hat.T,)))

plt.subplot(515)
plt.plot(t_eval,vals_lqr_new[:,0].T,label='x1')
plt.plot(t_eval,vals_lqr_new[:,1].T,label='x2')
plt.plot(t_eval,vals_lqr_new[:,2].T,label='x3')
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