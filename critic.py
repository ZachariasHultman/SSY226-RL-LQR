import numpy as np
from tools import kronecker
from tools import vech_to_mat_sym, sigma_fun
# equation 20 is the following function


def approx_update(x_hist,u_hist, W_c_hat, alpha_c, M, R, dt, n ,m):
  
    # u = np.atleast_2d(u)
    # x = np.atleast_2d(x)
    # print("x",x_hist.shape)
    # print("u",u_hist.shape)
    # print("u_prev", u_prev.shape)
    # print("x_prev", x_prev.shape)
    # print(x_hist[:,-1].reshape(2,1))

    U = np.concatenate((x_hist[:,-1].reshape(n,1).T, u_hist[:,-1].reshape(m,1).T),1).T

    U_prev = np.concatenate((x_hist[:,0].reshape(n,1).T, u_hist[:,0].reshape(m,1).T),1).T
    # print('INTERNAL U',U)
    # print('INTERNAL U_PREV',U_prev)

    sigma = sigma_fun(U, U_prev, n, m)
   
    # Compute new Weights
    # See equation below (e=...) to understand the need of the integral term.
    # The integral term is calculated by assumptions that self-defined matrices M and R are diagonal
    # and assumption that the integration is discrete with time step T and only two points of evaluation
    
    int_term=np.zeros(x_hist.shape[1])
    for k in range(x_hist.shape[1]):
        int_term[k] = x_hist[:,k].reshape(n,1).T @ M @ x_hist[:,k].reshape(n,1) + (u_hist[:,k].reshape(m,1).T @ R @ u_hist[:,k].reshape(m,1))

    # print(int_term.shape)
    # print(int_term)

    int_term=np.trapz(int_term,dx=dt)
    # print('int term',int_term)

    # Gör skillnad på T och dt. Implementera X_histry eller nåt så man går över längre tidssteg och större integral
    # if u.shape[0]<2:
    #     int_term = 0.5 * dt* ( (np.matmul(np.matmul(x.T, M), x) + u*R*u) +
    #                               (np.matmul(np.matmul(x_prev.T, M), x_prev) + u_prev*R*u_prev))
    # else:
    #     int_term = 0.5 * dt * ((np.matmul(np.matmul(x.T, M), x) + np.matmul(np.matmul(u.T, R), u)) +
    #                             (np.matmul(np.matmul(x_prev.T, M), x_prev) + np.matmul(np.matmul(u_prev.T, R), u_prev)))
    

    # Using integral RL gives error of (Bellman) value function as (eq.17 to eq.18)
    e = W_c_hat.T @ kronecker(U, U,n,m) + 0.5 *int_term - W_c_hat.T @ kronecker(U_prev, U_prev,n,m)
    # e=np.abs(e)
    print(e)
    # print(sigma)
    # print(sigma.T @ sigma)
    # br

    # Update of the critic approximation weights (Equation 20)

    # print((1 + np.matmul(sigma.T, sigma))**2)
    # W_c_hat_dot = -alpha_c * sigma / ((1 + sigma.T @ sigma)**2) * e.T
    W_c_hat_dot = -alpha_c * sigma * e.T

    # print('W_c_hat_dot',W_c_hat_dot)
    # br
    # print('INTERNAL SIGMA**2',np.matmul(sigma.T, sigma)) #This gets really big


    # W_c_tilde_dot = -alpha_c * np.matmul((np.matmul(sigma, sigma.T) / ((1 + np.matmul(sigma.T, sigma))**2)), W_c_tilde)
    # Q_bar_tilde = vech_to_mat_sym(W_c_tilde, n + m)
    # Q_xu_tilde = Q_bar_tilde[n:,:n].T

    return W_c_hat_dot #, W_c_tilde_dot, Q_xu_tilde


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

