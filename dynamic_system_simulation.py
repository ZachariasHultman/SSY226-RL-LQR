import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import critic
import actor

def func(t,y,sysfunc, L, m_p, M_p, g, F, f,b,n,m,x_prev, u_prev, alpha_c, alpha_a, M, R, T,explore):
    
    s = int(1 / 2 * ((n + m) * (n + m + 1)))
    x = np.expand_dims(y[:n], 1)
    W_a_hat = y[n:n+n]
 
    W_c_hat = y[n+n:n+n+s]
    W_c_tilde = y[n+n+s:]

    u = np.matmul(W_a_hat.T, x)+np.random.normal(0,np.abs(u_prev)*explore,m)
    
    u = np.atleast_2d(u)
    # print(u)
    x_dot = sysfunc(x, u, L, m_p, M_p, g, f, b)
     
    W_c_hat_dot, W_c_tilde_dot, Q_xu_tilde = critic.approx_update(x, x_prev, u, u_prev, W_c_hat, W_c_tilde, alpha_c, M, R, T, n, m)
    W_a_hat_dot, W_a_tilde_dot = actor.approx_update(x, Q_xu_tilde, W_a_hat, W_c_hat, n, m, alpha_a)
 


    
    states = x_dot
    states += [s for s in W_a_hat_dot]
    states += [s for s in W_c_hat_dot]
    states += [s for s in W_c_tilde_dot]


    return states

def cart_pendulum_sim(t ,x, L=1., m=1., M = 1., g=9.81, F=0, f=0,b=0):
    """
    x_1_dot = velocity
    x_2_dot = acceleration
    x_3_dot = angular velocity
    x_4_dot = angular acceleration
    """
    # print(x)
    x1, x2, x3, x4 = x
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


def diesel_lqr(t ,x, K, L=1., m=1., M = 1., g=9.81 , f=0 ,b=0):
    """
    Function for simulating a controlled cart pendulum linerized around theta=0
    x_1_dot = velocity
    x_2_dot = acceleration
    x_3_dot = angular velocity
    x_4_dot = angular acceleration

    returns: xdot as a list
    """
    # print(x)

    x1, x2, x3, x4, x5, x6 = x

    x_arr=np.array([x1,x2,x3,x4,x5,x6])
    u=-1*np.matmul(x_arr,K)

    A=np.array([[-0.4125, -0.0248, 0.0742, 0.0089, 0, 0],
                [101.5873, -7.2651, 2.7608, 2.8068, 0, 0],
                [0.0704, 0.0085, -0.0741, -0.089, 0, 0.02],
                [0.0878, 0.2672, 0, -0.3674, 0.0044, 0.3962],
                [-1.8414, 0.099, 0, 0, -0.0343, -0.0330],
                [0, 0, 0, -359, 187.5364, -87.0316]])

    B=np.array([[-0.0042 ,0.0064],
                [-1.0360, 1.5849],
                [0.0042, 0],
                [0.1261, 0],
                [0, -0.0168],
                [0, 0]])

    xdot=np.matmul(A,x)+np.matmul(B,u)            

    

    return xdot