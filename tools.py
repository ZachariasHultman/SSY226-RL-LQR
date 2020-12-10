import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import control as ctrl



def norm_error(true_value, approx_value):
    
    e = 1/(np.linalg.norm(true_value))*np.linalg.norm(approx_value-true_value)

    return e


def double_integrator_lin_lqr_gain(Q, R):

    A = np.array([[0, -1],
                 [0, -0.1]])

    B = np.array([0, 1]).T
    B = np.expand_dims(B, axis=1)

    K, P, E = ctrl.lqr(A, B, Q, R)

    K = np.array([K[0, 0], K[0, 1]])

    return K, P


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


def sigma_fun(U_curr, U_prev, n, m):
    """
    Help function to calculate sigma in critic weights equation
    Parameters: U_curr, which is [states; control signal] ([x;u]) concatinated at the current time step. array_like. Size N*Mx1
                U_prev, which is [states; control signal] ([x;u]) concatinated at the previous time step. array_like Size N*Mx1

    Out: sigma, array like. Size N^2xM^2
    """

    sigma_pt1 = kronecker(U_curr, U_curr, n, m)
    sigma_pt2 = kronecker(U_prev, U_prev, n, m)
    

    sigma = sigma_pt1 - sigma_pt2
    
    return sigma



def kronecker(A,B,n,m):
    # A=np.array([[1],[2],[3],[4]])
    # B=np.array([[1],[2],[3],[4]])
    k=A.shape[0]
    # print(k)
    # [1*1 , 2*2, 1*2, 1*3,2*3,3*3]
    # [x1**2, x2**2, x1x2, x1u, x2u, u**2]
    # [x1**2, x2**2, x1x2, x1u1, x1u2, x2u1,x2u2, u1u2 ,u1**2, u2**2]
    
    s = int(1 / 2 * ((n + m) * (n + m + 1)))
    C=np.zeros(s)

    for i in range(n):
        C[i]=A[i]*B[i]

    c=n
    for i in range(n+m-1):
        for j in range(i+1,n+m):
            C[c]=A[i]*B[j]
            c+=1

    for i in range(1,m+1):
        C[-i]=A[-i]*B[-i]
   

    return C.reshape(s,1)


def vech_to_mat_sym(a, n,m):
    """
    Takes a vector which represents the elements in a upper (or lower)
    triangular matrix and returns a symmetric (n x n) matrix.
    The off-diagonal elements are divided by 2.

    :param a: input vector of type np array and size n(n+1)/2
    :param n: dimension of symmetric output matrix A
    :return: symmetric matrix of type np array size (n x n)
    """
    s = int(1 / 2 * ((n + m) * (n + m + 1)))
    A = np.ndarray((n+m, n+m))

    for tmp in range(n):
        A[tmp,tmp]=a[tmp]
        
    c=m
    for j in range(n,n+m):
        for i in range(j,n+m):
            for tmp in range(s-c,s):
                A[i,i]=a[tmp]
                c-=1
                break
  
    c=n
    for j in range(n+m):
        for i in range(j+1,n+m):
            A[i,j]=a[c]/2
            A[j,i]=a[c]/2
            c+=1
    return A



def mat_to_vec_sym(A, n, m):
    s = int(1 / 2 * ((n + m) * (n + m + 1)))
    a = np.ndarray((s))
    c = 0
    # [1,4,3,5,6,7,8,11,9,12]
    
    for tmp in range(n):
        a[tmp]=A[tmp,tmp]
        
    c=m
    for j in range(n,n+m):
        for i in range(j,n+m):
            for tmp in range(s-c,s):
                a[tmp]=A[i,i]
                c-=1
                break
        
    c=n
    for j in range(n+m):
        for i in range(j+1,n+m):
            a[c]=A[i,j]*2
            c+=1
    return a



