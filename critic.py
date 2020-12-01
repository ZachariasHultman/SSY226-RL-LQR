import numpy as np
from tools import kronecker
from tools import vech_to_mat_sym, vech_to_mat, sigma_fun
# equation 20 is the following function


def approx_update(x, x_prev, u, u_prev, W_c_hat, W_c_tilde, alpha_c, M, R, T, n ,m):

    U = np.concatenate((x.T, u.T), 1).T

    U_prev = np.concatenate((x_prev.T, u_prev.T)).T
    print('INTERNAL U',U)


    print('INTERNAL U_PREV',U_prev)


    sigma = sigma_fun(U, U_prev, n, m)

    # Compute new Weights
    # See equation below (e=...) to understand the need of the integral term.
    # The integral term is calculated by assumptions that self-defined matrices M and R are diagonal
    # and assumption that the integration is discrete with time step T and only two points of evaluation
    
    if u.shape[0]<2:
        int_term = 0.5 * T * ((np.matmul(np.matmul(x.T, M), x) + u*R*u) +
                                np.matmul(np.matmul(x_prev.T, M), x_prev) + u_prev*R*u_prev)
    else:
        int_term = 0.5 * T * (np.matmul(np.matmul(x.T, M), x) + np.matmul(np.matmul(u.T, R), u) +
                                np.matmul(np.matmul(x_prev.T, M), x_prev) + np.matmul(np.matmul(u_prev.T, R), u_prev))
    

    # Using integral RL gives error of (Bellman) value function as (eq.17 to eq.18)
    e = np.matmul(W_c_hat.T, kronecker(U, U,n,m)) + int_term - np.matmul(W_c_hat.T, kronecker(U_prev, U_prev,n,m))
    

    # Update of the critic approximation weights (Equation 20)

    # print((1 + np.matmul(sigma.T, sigma))**2)
    W_c_hat_dot = -alpha_c * sigma / ((1 + np.matmul(sigma.T, sigma))**2) * e.T
    # print('INTERNAL SIGMA**2',np.matmul(sigma.T, sigma)) This gets really small


    W_c_tilde_dot = -alpha_c * np.matmul((np.matmul(sigma, sigma.T) / ((1 + np.matmul(sigma.T, sigma))**2)), W_c_tilde)
    Q_bar_tilde = vech_to_mat_sym(W_c_tilde, n + m)
    Q_xu_tilde = Q_bar_tilde[n:,:n].T

    return W_c_hat_dot, W_c_tilde_dot, Q_xu_tilde


def Q_uu(n,m,W_hat):

    # Extract Q_uu from Wc on vector form from
    q_vec = W_hat[int(n * (n + 1) / 2 + 1 + n * m):int((n + m) * (n + m + 1) / 2)]
    # Reshape to matrix form
    q_uu = vech_to_mat_sym(q_vec, m)
    return q_uu


def Q_xu(n,m,W_hat):

    # Extract Q_xu from Wc on vector form from
    q_vec = W_hat[int(n * (n + 1) / 2 + 1):int(n * (n + 1) / 2 + n * m)]
    # Reshape to matrix form
    q_xu = vech_to_mat(q_vec, n, m)
    return q_xu

