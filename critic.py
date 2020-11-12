import numpy as np

#equation 20 is the following function

class critic():
    def __init__(self ,n ,m ,alpha):
        self.W = np.ones(0.5*(n+m)*(n+m+1),1)
        self.alpha = alpha

    def approx_update(self, x, x_prev, u, u_prev, M, R, T, sigma):
        U = np.concatenate(x, u)
        U_prev = np.concatenate(x_prev, u_prev)
        tol = 0.01  # Tolerance
        error = tol+1  # Init error larger than tol

        while error > tol:
            # Save old weights

            W_old = self.W

            # Compute new Weights
            #See equation below (e=...) to understand the need of the integral term.
            #The integral term is calculated by assumptions that self-defined matrices M and R are diagonal
            #and assumption that the integration is discrete with time step T and only two points of evaluation

            int_term = 0.5 * T * (np.matmul(np.matmul(x.T, M), x) + np.matmul(np.matmul(u.T, R), u) +
                                  np.matmul(np.matmul(x_prev.T, M), x_prev) + np.matmul(np.matmul(u_prev.T, R), u_prev))

            #Using integral RL gives error of (Bellman) value function as (eq.17 to eq.18)

            e = self.W.T * np.kron(U, U) + int_term - self.W.T * np.kron(U_prev, U_prev)

            #Update of the critic approximation weights (Equation 20)

            self.W = -self.alpha * sigma / ((1 + np.matmul(sigma.T, sigma))**2) * e.T

            # Calculates the error as 2-norm of the difference between the new and old W-matrix.

            error = np.linalg.norm(self.W-W_old, ord=2)