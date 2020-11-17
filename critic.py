import numpy as np
from tools import kronecker

# equation 20 is the following function


class critic():
    def __init__(self, n, m, alpha):
        s=(int(1/2*((n+m)*(n+m+1))),1)
        self.W = np.ones(s)
        self.error = np.Inf
        self.alpha = alpha
        self.n = n
        self.m = m

    def approx_update(self, x, x_prev, u, u_prev, M, R, T):

        U = np.concatenate((x.T, u.T)).T
        U_prev = np.concatenate((x_prev.T, u_prev.T)).T
        print(U)
        tol = 0.01  # Tolerance
        error = tol+1  # Init error larger than tol
        W_old=self.W
        sigma = self.sigma_fun(u, u_prev)
        # while error > tol:
            

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
        e = np.matmul(self.W.T, kronecker(U, U,self.n,self.m)) + int_term - np.matmul(self.W.T, kronecker(U_prev, U_prev,self.n,self.m))

        print('eureka')

        # Update of the critic approximation weights (Equation 20)

        self.W = -self.alpha * sigma / ((1 + np.matmul(sigma.T, sigma))**2) * e.T

        # Calculates the error as 2-norm of the difference between the new and old W-matrix.
        error_weights = W_old - self.W

        error = -self.alpha * (np.matmul(sigma, sigma.T) / ((1 + np.matmul(sigma.T, sigma))**2))* error_weights
        error_diff = np.linalg.norm(error - self.error, ord=2)
        self.error = error

        # Save old weights
        W_old = self.W


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

    def sigma_fun(self, U_curr, U_prev):
        """
        Help function to calculate sigma in critic weights equation
        Parameters: U_curr, which is [states; control signal] ([x;u]) concatinated at the current time step. array_like. Size NxM
                    U_prev, which is [states; control signal] ([x;u]) concatinated at the previous time step. array_like Size NxM

        Out: sigma, array like. Size N^2xM^2
        """

        sigma_pt1 = np.kron(U_curr, U_curr)
        sigma_pt2 = np.kron(U_curr, U_curr)
        sigma = np.kron(sigma_pt1, sigma_pt2)
        return sigma
