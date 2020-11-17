import numpy as np

# equation 20 is the following function


class critic():
    def __init__(self, n, m, alpha):
        self.W = np.ones(0.5*(n+m)*(n+m+1), 1)
        self.error = np.Inf
        self.alpha = alpha
        self.n = n
        self.m = m

    def approx_update(self, x, x_prev, u, u_prev, M, R, T, sigma):
        U = np.concatenate(x, u)
        U_prev = np.concatenate(x_prev, u_prev)
        tol = 0.01  # Tolerance
        error = tol+1  # Init error larger than tol

        while error_diff > tol:
            # Save old weights

            W_old = self.W

            # Compute new Weights
            # See equation below (e=...) to understand the need of the integral term.
            # The integral term is calculated by assumptions that self-defined matrices M and R are diagonal
            # and assumption that the integration is discrete with time step T and only two points of evaluation

            int_term = 0.5 * T * (np.matmul(np.matmul(x.T, M), x) + np.matmul(np.matmul(u.T, R), u) +
                                  np.matmul(np.matmul(x_prev.T, M), x_prev) + np.matmul(np.matmul(u_prev.T, R), u_prev))

            # Using integral RL gives error of (Bellman) value function as (eq.17 to eq.18)

            e = np.matmul(self.W.T, np.kron(U, U)) + int_term - np.matmul(self.W.T, np.kron(U_prev, U_prev))

            # Update of the critic approximation weights (Equation 20)

            self.W = -self.alpha * sigma / ((1 + np.matmul(sigma.T, sigma))**2) * e.T

            # Calculates the error as 2-norm of the difference between the new and old W-matrix.
            error_weights = W_old - self.W
            error = -self.alpha * np.matmul(np.matmul(sigma, sigma.T) / ((1 + np.matmul(sigma.T, sigma))**2), error_weights)
            error_diff = np.linalg.norm(error - self.error, ord=2)
            self.error = error


    def Q_uu(self):
        n = self.n
        m = self.m
        # Extract Q_uu from Wc on vector form from
        q_vec = self.W[n * (n + 1) / 2 + 1 + n * m:(n + m) * (n + m + 1) / 2]
        # Reshape to matrix form
        q_uu = q_vec.reshape((m, m))
        return q_uu

    def Q_ux(self):
        n = self.n
        m = self.m
        # Extract Q_xu from Wc on vector form from
        q_vec = self.W[n * (n + 1) / 2 + 1:n * (n + 1) / 2 + n * m]
        # Reshape to matrix form
        q_xu = q_vec.reshape((self.n, self.m))
        # Transpose to get Q_ux
        q_ux = q_xu.T
        return q_ux
