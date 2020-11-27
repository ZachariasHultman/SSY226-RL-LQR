##This document implements the QSA off policy iteration
import numpy as np
from tools import kronecker

##Define known case for testing
A = np.array([[0, -1],[0, -0.1]])
B = np.array([0,1])
M = np.eye(2)
R = 10*eye(1)

#Define initial states and inputs
x = np.array([0, 0])
u = np.array([0])

U = np.concatenate(x, u)

n = np.size(x)  #Sates size
m = np.size(u)  #Input size
#Define c(x,u):

c = np.matmul(x.T,np.matmul(M,x)) + np.matmul(u.T,np.matmul(R,u))
d = np.copy(c)

#Define Si: #kronecker(A,B,n,m)
Si = kronecker(U, U, m, n)

#Define optimization parameter
theta = np.zeros(np.size(Si))

#Initialize K #In paper page 5, K = [5, 1]
K = np.array([-1, 0])

#Define phi: = K*x
phi = np.matmul(K,x)

#Define eq 24:

#Define zeta:
U1 = np.concatenate(x,phi)
zeta_pt1 = kronecker(U1, U1, n, m)
zeta_pt3 = diff_func(U1, n, m)
zeta = zeta_pt1 - Si + zeta_pt3

#Define b(t):


#Eq 23:



def diff_func(U, n, m):

    for i in range(n):
        d[i] = 2*U[i]

    for i in range(n + m - 1):
        for j in range(i, n + m - 1):
            d[i + n] = U[i] + U[j]

    for i in range(1, m + 1):
        d[-i] = 2*U[-i]

    return d
#################Testing site
# import sympy as sym
# import numpy as np
#
# x = sym.Symbol('x', 'y')
# z = np.array([x**5 *x**2, x**4 * x**2])
# d = sym.diff(z)
# print(d)