import numpy as np
from tools import vech_to_mat_sym


def approx_update(x, W_a_hat, W_c_hat, n, m, alpha_a):
    # print(W_c_hat)
    # W_c_hat=[1 ,2 ,3 ,4 ,5 ,6]
    Q_bar = vech_to_mat_sym(W_c_hat, n,m)
    
    # print(Q_bar)
    # Q_bar=Q_bar*2
    Q_bar_ux = Q_bar[n:,:n]
    Q_bar_xu = Q_bar_ux.T
    # Q_bar_xu= W_c_hat[int(n*(n+1)/2 +1)-1:int(n*(n+1)/2 +n*m)]
    # Q_bar_xu= vech_to_mat(Q_bar_xu,n,m)
    # Q_bar_ux=Q_bar_xu.T
    # print(Q_bar_ux)

    Q_bar_uu = Q_bar[n:,n:]
    # Q_bar_uu=W_c_hat[int(n*(n+1)/2 +1+n*m)-1:int((n+m)*(n+m+1)/2)]
    # Q_bar_uu = vech_to_mat_sym(Q_bar_uu,m)

    # compute actor error
    e_a = W_a_hat.T @ x - np.linalg.pinv(Q_bar_uu) @ Q_bar_ux @ x
    # print(W_a_hat)
    # print(np.linalg.pinv(Q_bar_uu) @ Q_bar_ux)
    # print(e_a)
    # br
    W_a_hat_dot = -alpha_a*x @ e_a.T
    
    # Calculates the error as 2-norm of the difference between the new and old W-matrix.
    # br
    # W_a_tilde = -np.matmul(Q_bar_xu,np.linalg.pinv(Q_bar_uu))-W_a_hat

    # W_a_tilde_dot = -alpha_a*np.matmul(np.matmul(x, x.T),W_a_tilde)-alpha_a*np.matmul(np.matmul(np.matmul(x, x.T),Q_xu_tilde),np.linalg.pinv(Q_bar_uu))

    return W_a_hat_dot #, W_a_tilde_dot



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
