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
    sigma_pt2=np.kron(U_prev,U_prev)
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


def vech_to_mat_sym(a, n):
    """
    Takes a vector which represents the elements in a upper (or lower)
    triangular matrix and returns a symmetric (n x n) matrix.
    The off-diagonal elements are divided by 2.

    :param a: input vector of type np array and size n(n+1)/2
    :param n: dimension of symmetric output matrix A
    :return: symmetric matrix of type np array size (n x n)
    """

    A = np.ndarray((n, n))

    c = 0
    for j in range(n):
        for i in range(j, n):

            if i == j:
                A[i, j] = a[c]
                A[j, i] = a[c]
            else:
                A[i, j] = a[c]/2
                A[j, i] = a[c]/2
            c += 1

    return A


def vech_to_mat(a, n, m):
    """
    Takes a vector a and stacks the elements in a (n x m) matrix

    :param a: vector of type np array of size nm
    :param n: rows in output matrix
    :param m: columns in output matrix
    :return: (n x m) matrix of type np array
    """

    A = np.ndarray((n, m))
    print(a)
    print(n)
    print(m)
    for j in range(m):
        for i in range(n):
            A[i, j] = a[i+n*j]
    return A
