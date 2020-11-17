import numpy as np



def approx_update(x, Q_xu_tilde, W_a_hat, n, m, alpha_a):


    # compute actor error
    e_a = np.matmul(W_a_hat.T, x) + np.matmul(np.matmul(np.linalg.inv(Q_uu(n,m,W_a_hat)), Q_xu(n,m,W_a_hat)), x)

    W_a_hat_dot = -alpha_a*np.matmul(x, e_a.T)
    # Calculates the error as 2-norm of the difference between the new and old W-matrix.
    W_a_tilde = -np.matmul(Q_xu,np.linalg.inv(Q_uu))-W_a_hat
    W_a_tilde_dot = -alpha_a*np.matmul(np.matmul(x, x.T),W_a_tilde)-alpha_a*np.matmul(np.matmul(np.matmul(x, x.T),Q_xu_tilde),np.linalg.inv(Q_uu))

    return W_a_hat_dot, W_a_tilde_dot

def Q_uu(n,m,W_hat):

    # Extract Q_uu from Wc on vector form from
    q_vec = W_hat[int(n * (n + 1) / 2 + 1 + n * m):int((n + m) * (n + m + 1) / 2)]
    # Reshape to matrix form
    q_uu = q_vec.reshape((m, m))
    return q_uu

def Q_xu(n,m,W_hat):

    # Extract Q_xu from Wc on vector form from
    q_vec = W_hat[int(n * (n + 1) / 2 + 1):int(n * (n + 1) / 2 + n * m)]
    # Reshape to matrix form
    q_xu = q_vec.reshape((n, m))
    return q_xu
