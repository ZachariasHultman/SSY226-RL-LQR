import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import control as ctrl
from dynamic_system_simulation import cart_pendulum_sim_lqr
from dynamic_system_simulation import cart_pendulum_sim_lqr2
from dynamic_system_simulation import func, double_integrator_with_friction, double_integrator_with_friction2, double_integrator_with_friction_ODE, double_integrator_with_friction2_ODE, test_sys_ODE, test_sys2_ODE, test_sys2
from tools import cart_pendulum_lin_lqr_gain, double_integrator_lin_lqr_gain, norm_error, mat_to_vec_sym, vech_to_mat_sym
from AnimationFunction import animationfunction
import critic
import actor
global t_ac
global x_ac
global u_hist


T = 0.05  
dt=0.001 # delta t [s]
t_span =[0, 40]  # Time span for simulation
t_eval = np.linspace(t_span[0],t_span[1],int(1/dt))  # Time span for simulation
# ---------------------------------------------------------------------------------
n = 3
m = 2
sys_func=test_sys_ODE
A =np.array([[-1.01887, -0.90506, -0.00215],
     [0.82225, -1.07741, -0.17555],
     [0, 0, -1]])

B = np.array([[0, 0, 1], [1, 1, 1]]).T

M = 2*np.eye(n)
R = 1*np.eye(m)
x_init=[0.1, 0.1, -0.1]

yay = [0.1000,    0.1000,   -0.1000,    0.9821,    0.5783,    0.2344,    0.8106,    0.4513,    0.2500,    0.9554,    0.1427,    0.5126,    0.9719,    0.6483,    0.6147,    0.4697,    0.5778,    0.9113,    0.3762,    0.2736, 0.2288,   0.4446,  0.4235,   0.6275,0]
W_c_hat =yay[3:18]
W_a_hat = yay[18:24]
int_term=yay[-1]
states=x_init
states += [s for s in W_a_hat]
states += [s for s in W_c_hat]
states += [int_term]

# ---------------------DOUBLE INTEGRAL-----------------------------------------------------------
# n = 2
# m = 1
# sys_func=double_integrator_with_friction_ODE
# A = np.array([[0, -1],
#                 [0, -0.1]])
# B = np.array([[0],[1]])
# M = np.array([[ 1,         0],
#             [ 0,         1]])
# R = np.array([0.1])
# R= R.reshape(m,m)
# x_init=[0, 1]  # Initial state

# -------------------------------------------------------------------------------------

# K_lqr, P = double_integrator_lin_lqr_gain(M, R)
K_lqr, P, E =ctrl.lqr(A, B, M, R)
# P, L_CARE, K_lqr =ctrl.care(A, B, M)
P_CARE, L_CARE, U_CARE =ctrl.care(A, B, M)
P_DARE, L_DARE, U_DARE =ctrl.dare(A, B, M,R)
print(K_lqr)

# the second has to altered if other system is used
# vals_lqr = integrate.solve_ivp(sys_func, t_span, x_init, t_eval=t_eval, args=args)
vals_lqr=integrate.odeint(sys_func, [0.1, 0.1, -0.1],  t_eval, args=((K_lqr,)))
# vals_lqr=integrate.odeint(sys_func, [0, 1] ,  t_eval, args=((K_lqr,)))

# Simulation with Actor-Critic
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
print('W_c_opt',W_c_opt)
#ekv 28
delta=1
alpha_a_upper=(1/delta*np.max(np.linalg.eigvals(np.linalg.inv(np.atleast_2d(R)))))*((2*np.min(np.linalg.eigvals(M+(Q_xu@ np.linalg.inv(np.atleast_2d(R))@ Q_xu.T))))-np.max(np.linalg.eigvals(Q_xu@Q_xu.T)))
print('alpha_a_upper',alpha_a_upper)
W_a_opt= (-np.linalg.pinv(R)@B.T@P).T 
print('W_a_opt',W_a_opt)

# if double integral is used insted
# W_a_hat=(W_a_opt+1).tolist()
# W_c_hat=W_c_opt+1
# states=x_init
# states += [s[0] for s in W_a_hat]
# states += [s for s in W_c_hat]
# states += [0]

# -----------------------------------------------------------------------------
alpha_c = 50
alpha_a = 2
explore=1
s = int(1 / 2 * ((n + m) * (n + m + 1)))
args_ac = (A,B, n, m, alpha_c, alpha_a, M, R, T, explore,dt,t_span[1])
vals= integrate.solve_ivp(func, t_span, states, args=args_ac)
W_a_hat=vals.y[-1,n:n+n*m]

vals_lqr_new=integrate.odeint(sys_func, [0.1, 0.1, -0.1],  t_eval, args=((np.asarray(W_a_hat).reshape(n,m).T,)))
# vals_lqr_new=integrate.odeint(sys_func, [0, 1],  t_eval, args=((np.asarray(W_a_hat).reshape(n,m).T,)))


e = norm_error(W_a_opt, np.asarray(-W_a_hat).reshape(n,m))
print('W_a error',e)

plt.figure()
# Plotting controlled linear system
plt.subplot(311)
plt.plot(t_eval,vals_lqr[:,0].T,label='x1')
plt.plot(t_eval,vals_lqr[:,1].T,label='x2')
plt.plot(t_eval,vals_lqr[:,2].T,label='x3')
plt.legend(loc="upper left")

plt.subplot(312)
plt.plot(t_eval,vals_lqr_new[:,0].T,label='x1')
plt.plot(t_eval,vals_lqr_new[:,1].T,label='x2')
plt.plot(t_eval,vals_lqr_new[:,2].T,label='x3')
plt.legend(loc="upper left")


# Plotting controlled linear system
plt.subplot(313)
plt.plot(vals.y,label='x1')
# plt.plot(t_ac,x_ac[1,:].T,label='x2')
# plt.plot(t_ac,x_ac[2,:].T,label='x3')
plt.legend(loc="upper left")
plt.show()


