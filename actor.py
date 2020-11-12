import numpy as np

class actor():
    def __init__(self, n, m, alpha):
        self.W = np.ones((n, m))
        self.alpha = alpha

    def approx_update(self, Q_uu, Q_xu, x, Q_xu_tilde):

        tol = 0.01  # Tolerance
        error = tol+1  # Init error larger than tol

        while error > tol:
            # Save old weights
            W_old = self.W
            # Compute new Weights
            e_a = np.matmul(self.W.T, x) + np.matmul(np.matmul(np.linalg.inv(Q_uu), Q_xu), x)

            self.W = -self.alpha*np.matmul(x, e_a.T)
            # Calculates the error as 2-norm of the difference between the new and old W-matrix.
            error_pt1=-self.alpha*np.matmul(np.matmul(x, x.T),self.W)-self.alpha*np.matmul(np.matmul(np.matmul(x, x.T),Q_xu_tilde),np.linalg.inv(Q_uu))

            error = np.linalg.norm(self.W-W_old, ord=2)

def Q_uu(self):
    n = self.n
    m = self.m
    # Extract Q_uu from Wc on vector form from
    q_vec = self.W[n * (n + 1) / 2 + 1 + n * m:(n + m) * (n + m + 1) / 2]
    # Reshape to matrix form
    q_uu = q_vec.reshape((m, m))


def Q_xu(self):
    n = self.n
    m = self.m
    # Extract Q_xu from Wc on vector form from
    q_vec = self.W[n * (n + 1) / 2 + 1:n * (n + 1) / 2 + n * m]
    # Reshape to matrix form
    q_xu = q_vec.reshape((self.m, self.m))
    return q_xu
