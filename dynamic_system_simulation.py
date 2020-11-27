import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import critic
import actor

#def func(t,y,sysfunc, L, m_p, M_p, g, F, f,b,n,m,x_prev, u_prev, alpha_c, alpha_a, M, R, T,explore):
def func(t,y,sysfunc, n, m, x_prev, u_prev, alpha_c, alpha_a, M, R, T,explore):
    s = int(1 / 2 * ((n + m) * (n + m + 1)))
    x = np.expand_dims(y[:n], 1)
    W_a_hat = y[n:n+n*m]
 
    W_c_hat = y[n+n:n+n+s]
    W_c_tilde = y[n+n+s:]

    u = np.matmul(W_a_hat, x)+np.random.normal(0, explore, m)
    
    u = np.atleast_2d(u)
    # print(u)
    x_dot = sysfunc(x, u)
    
    W_c_hat_dot, W_c_tilde_dot, Q_xu_tilde = critic.approx_update(x, x_prev, u, u_prev, W_c_hat, W_c_tilde, alpha_c, M, R, T, n, m)
    W_a_hat_dot, W_a_tilde_dot = actor.approx_update(x, Q_xu_tilde, W_a_hat, W_c_hat, n, m, alpha_a)

    
    states = x_dot
    states += [s for s in W_a_hat_dot]
    states += [s for s in W_c_hat_dot]
    states += [s for s in W_c_tilde_dot]


    return states


def double_integrator_with_friction2(x, u):

    x1, x2 = x

    x_1_dot = -x2
    x_2_dot = -0.1*x2 + u

    return [x_1_dot, x_2_dot]


def double_integrator_with_friction(t, x, K):

    x1, x2 = x

    u = -np.matmul(K, x)

    x_1_dot = -x2
    x_2_dot = -0.1*x2 + u


    return [x_1_dot, x_2_dot]

def cart_pendulum_sim_lqr(t ,x, K, L=1., m=1., M = 1., g=9.81 , f=0 ,b=0):
    """
    Function for simulating a controlled cart pendulum linerized around theta=0
    x_1_dot = velocity
    x_2_dot = acceleration
    x_3_dot = angular velocity
    x_4_dot = angular acceleration

    returns: xdot as a list
    """
    # print(x)
    

    x1, x2, x3, x4 = x
    u=np.array([x1,x2,x3,x4])
    F=-1*np.matmul(u,K)


    x_2_dot_nomi = (-m*g*np.sin(x3)*np.cos(x3) +
                    m*L*x4*x4*np.sin(x3) +
                    f*m*x4*np.cos(x3)+F)-x2*b

    x_2_dot_denomi = (M + (1 - np.cos(x3) * np.cos(x3)) * m)

    x_4_dot_nomi = ((M+m)*(g*np.sin(x3)-f*x4) -
                    (L*m*x4*x4*np.sin(x3)+F) * np.cos(x3))

    x_4_dot_denomi = L*x_2_dot_denomi

    x_1_dot = x2
    x_2_dot = x_2_dot_nomi/x_2_dot_denomi
    x_3_dot = x4
    x_4_dot = x_4_dot_nomi/x_4_dot_denomi ###

    return [x_1_dot, x_2_dot, x_3_dot, x_4_dot]


def cart_pendulum_sim_lqr2(x, F, L=1., m=1., M=1., g=9.81, f=0, b=0):
    """
    Function for simulating a controlled cart pendulum linerized around theta=0
    x_1_dot = velocity
    x_2_dot = acceleration
    x_3_dot = angular velocity
    x_4_dot = angular acceleration

    returns: xdot as a list
    """
    # print(x)

    x1, x2, x3, x4 = x


    x_2_dot_nomi = (-m * g * np.sin(x3) * np.cos(x3) +
                    m * L * x4 * x4 * np.sin(x3) +
                    f * m * x4 * np.cos(x3) + F) - x2 * b

    x_2_dot_denomi = (M + (1 - np.cos(x3) * np.cos(x3)) * m)

    x_4_dot_nomi = ((M + m) * (g * np.sin(x3) - f * x4) -
                    (L * m * x4 * x4 * np.sin(x3) + F) * np.cos(x3))

    x_4_dot_denomi = L * x_2_dot_denomi

    x_1_dot = x2
    x_2_dot = x_2_dot_nomi / x_2_dot_denomi
    x_3_dot = x4
    x_4_dot = x_4_dot_nomi / x_4_dot_denomi  ###

    return [x_1_dot, x_2_dot, x_3_dot, x_4_dot]