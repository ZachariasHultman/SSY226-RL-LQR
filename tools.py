import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import control as ctrl


def cart_pendulum_lin_lqr_gain(L, m, M, g, f, b, Q, R):
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


def kronecker(A,B,n,m):

    k=A.shape[0]
    s=(int(1/2*((k)*(k+1))),1)
    C=np.ones(s)

    for i in range(n):
        C[i]=A[i]*B[i]

    for i in range(n+m-1):
        for j in range(i,n+m-1):
            C[i+n]=A[i]*B[j]

    for i in range(1,m+1):
        C[-i]=A[-i]*B[-i]



    return C

def vech_to_mat(a):
    n = int(-1/2 + np.sqrt(1/4+2*len(a)))

    A=np.zeros((n,n))
    for i in range(len(a)):
        for j in range(i,len(a)):
            A[i,j]=a[j]
            A[j,i]=a[j]

    return A
