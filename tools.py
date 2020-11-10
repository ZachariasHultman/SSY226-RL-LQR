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