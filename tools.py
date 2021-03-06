import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import control as ctrl


def cart_pendulum_lin_lqr_gain(L, m, M, g, f, b):
    """
    Function that calculates the lqr gain K for a linerized cart pendulum around theta=0.

    In: L: pendulum length
        m: pendulum mass
        M: cart mass
        g: gravity
        f: friction for pendulum
        b: friction for cart

    Returns: lqr gain K as np.array
    """
    
    Q=np.array([[ 1,         0,                 0,                  0],
            [ 0,         1,                 0,                  0],
            [ 0,         0,                 10,                  0],
            [ 0,         0,                 0,                  100]])

    R=0.01

    A=np.array([[ 0,         1,                 0,                  0],
                [ 0, -b/M,          -(g*m)/M,            (f*m)/M],
                [ 0,         0,                 0,                  1],
                [ 0,         0, (g*(M + m))/(L*M), -(f*(M + m))/(L*M)]])

    B=np.array([0, 1/M, 0, -1/(M*L)]).T
    B=np.expand_dims(B, axis=1)


    K ,S, E =ctrl.lqr(A, B, Q, R)

    K=np.array([K[0,0],K[0,1],K[0,2],K[0,3]])


    return K



    def sigma_fun(U_curr,U_prev):
        """
        Help function to calculate sigma in critic weights equation
        Parameters: U_curr, which is [states; control signal] ([x;u]) concatinated at the current time step. array_like. Size NxM
                    U_prev, which is [states; control signal] ([x;u]) concatinated at the previous time step. array_like Size NxM

        Out: sigma, array like. Size N^2xM^2
        """

        sigma_pt1=np.kron(U_curr,U_curr)
        sigma_pt2=np.kron(U_curr,U_curr)
        sigma=np.kron(sigma_pt1,sigma_pt2)
    return sigma