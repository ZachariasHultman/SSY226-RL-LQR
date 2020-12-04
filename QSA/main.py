# ##This document implements the QSA off policy iteration
# import numpy as np
# # from tools import kronecker
# import time
# import matplotlib.pyplot as plt
# import pandas as pd
#
# #Helper functions:
#
# def readfile_func(filename):
#     '''This function gets the offline values of time, states and input from given csv file ---> filename'''
#     datafile = pd.read_csv('{}.csv'.format(filename))
#     print(datafile.head())
#     t = datafile['t']
#     x1 = datafile['x1']
#     x2 = datafile['x2']
#     u = datafile['u']
#     return t, x1, x2, u
#
# def kronecker(A,B,n,m):
#
#     k=A.shape[0]
#     s=(int(1/2*((k)*(k+1))),1)
#     C=np.ones(s)
#
#     for i in range(n):
#         C[i]=A[i]*B[i]
#
#     for i in range(n+m-1):
#         for j in range(i,n+m-1):
#             C[i+n]=A[i]*B[j]
#
#     for i in range(1,m+1):
#         C[-i]=A[-i]*B[-i]
#
#     return C
#
# def diff_Si(U, n, m):
#     s = 0.5*(len(U)*(len(U)+1))
#     s = np.int(s)
#     d = np.zeros(s)
#
#     for i in range(n):
#         d[i] = 2*U[i]
#
#     for i in range(n + m - 1):
#         for j in range(i, n + m - 1):
#             d[i + n] = U[i] + U[j]
#
#     for i in range(1, m + 1):
#         d[-i] = 2*U[-i]
#
#     return d
#
# def diff_d(x,u,M,R):
#     print("Mx",np.matmul(R,u))
#     diff_d = 2*np.matmul(M,x) + 2* np.matmul(R,u)
#     return diff_d
#
# def cost_func(x,u,M,R):
#     cost = np.matmul(x.T,np.matmul(M,x)) + np.matmul(u.T,np.matmul(R,u))
#     return cost
#
# def Q_func(d, Si, theta):
#     Q = d + np.matmul(theta.T, Si)
#
#     return Q
#
# def eps_func(Q, Qscore, c, zeta, b, a, theta, M, R, x, phi, n, m):
#     dQscore = diff_d(x, phi ,M,R) + np.matmul(theta.T,diff_Si(np.concatenate(x,phi), m, n)) + np.matmul(diff_theta(zeta, b, a, theta),kronecker(np.concatenate(x,phi), np.concatenate(x,phi), n, m))
#     eps = -Q + Qscore + c + dQscore
#
#     return eps
#
# def diff_theta(zeta, b, a, theta, G_hat_inv):
#     dtheta_pt1 = (np.matmul(zeta.T, theta) + b)
#     dtheta = -a * np.matmul(G_hat_inv, dtheta_pt1*zeta)
#     return dtheta
#
# def compute_u(Ke, x, t):
#     q = 24 #Choosen in the paper as number of sinusoids
#     zeta_t = 0
#     freq = np.random.uniform(0,50,q)#np.random.choice(np.linspace(0, 1000), q)    # Frequency (0-50 rad/s)
#     phase = np.random.uniform(0,2*np.pi,q)# np.random.choice(np.linspace(0, 2*np.pi), q)    # Phase (0-1 rad)
#     a = np.random.uniform(0,10,q) #np.random.normal(q,1)                           # Amplitude (0-1)
#     for i in range(q):      # Compute sinusoidal noise to be introduced with input
#         zeta_t += a[i]*np.sin(freq[i]*t + phase[i])
#
#     u = np.matmul(Ke,x) + zeta_t
#     # u = u.reshape(1,1)#np.atleast_1d(u)
#
#     return u
#
# def G_inv_func(time_record, zeta_record, s):
#     integrand = np.zeros((s, s))
#     for i in range(1,len(time_record)-1):
#         integrand += (time_record[i] - time_record[i-1]) * (np.matmul(np.array(zeta_record[i-1:i]).reshape(s,1),np.array(zeta_record[i-1:i]).reshape(1,s)) + np.matmul(np.array(zeta_record[i:i+1]).reshape(s,1),np.array(zeta_record[i:i+1]).reshape(1,s))) / 2
#     G_hat = integrand / time_record[-1]
#     return np.linalg.pinv(G_hat)
#
# def double_integrator_with_friction_noise(t, x, K):
#     x1, x2 = x
#     u = compute_u(K,x,t)
#
#
#     x_1_dot = x2    #-x2
#     x_2_dot = -0.1 * x2 + u
#
#     return [x_1_dot, x_2_dot]
# #####============STRUCTURE OF CODE==============
# """"
# 1. Initialize all variables outside
# 2. While loop till theta - dtheta converges
# 3. counter of N = N + 1
# """
# # run simulation
# # from dynamic_system_simulation import cart_pendulum_sim_lqr
# from scipy import integrate
# filename = "offlinedata"
# [t, x1, x2, u] = readfile_func(filename)
#
# M = 0.5  # cart mass
# m = 0.2  # pendulum mass
# g = 9.81  # gravity
# L = 1  # pendulum length
# f = 0.1  # friction
# b= 0.5 #friction for cart
# F = 1  # control input [N]
# # Ke = -np.array([-1, 1.6349])
# Ke = -np.array([3, 2])
#
# x_init_lqr=[1, 0]  # Initial state. pos, vel, theta, thetadot for linearized system
# t_span =[0, 10]  # Time span for simulation
# t_eval =np.linspace(t_span[0],t_span[1],100)  # Time span for simulation
# args_lqr = (Ke,) #arguments for controlled linear system
#
# print("Start")
# vals_lqr = integrate.solve_ivp(double_integrator_with_friction_noise, t_span, x_init_lqr, args=args_lqr, t_eval=t_eval)
# print("Done")
#
# time_offline = vals_lqr.t
# x_offline = vals_lqr.y
# u_plot = np.matmul(Ke,x_offline)
# print(u_plot.shape)
# plt.plot(time_offline, x_offline[0,:], label = 'x1')
# plt.plot(time_offline, x_offline[1,:], label = 'x2')
# plt.plot(time_offline, u_plot[:], label = 'u')
# plt.legend(loc="upper left")
# plt.show()
# print("shape u_offline", time_offline.shape)
#
# #Initialize
# ##Define known case for testing
# A = np.array([[0, -1],[0, -0.1]])
# B = np.array([0,1])
# M = np.eye(2)
# R = 10*np.eye(1)
#
# g = 1.5  #Authors experimented with different values
#
# N = 1  #Defines number of iterations
#
# #Initialize K #In paper page 5, K = [5, 1]
# K = np.array([-1, 0])
# Ke = np.array([-1, -2])
#
# #Initialize states, input and parameters
# # Start the simulation time (we can look for better place)
# start = time.time()
# t = 0
# t_prev = 0
# #Define initial states and inputs
# x = np.array([0.0, 0.0])
# u = compute_u(Ke, x, t)
#
#
#
# U = np.concatenate((x, u), axis=0)
#
# s = 0.5*(len(U)*(len(U)+1))     # Size of kronecker array
# s = np.int(s)
#
#
# n = np.size(x)  #Sates size
# m = np.size(u)  #Input size
#
# phi = np.matmul(K,x)
# phi = np.atleast_1d(phi)
#
# size_theta = np.size(kronecker(U, U, m, n))   #Just used for gettin size for theta
#
# theta = np.zeros(size_theta)
# theta = theta.reshape(size_theta,1)
# dtheta = np.ones(size_theta)
# dtheta = dtheta.reshape(size_theta,1)
# d_prev = cost_func(x,u,M,R) - 10
# zeta_pt1_prev = kronecker(U, U, n, m)
#
# theta_record = []
# phi_record = []
# time_record = []
# Q_all = []
# zeta_record = []
#
# theta_record.append(theta)
# time_record.append(t)
# tol = 1e-3
#
# d_policy_off_prev = cost_func(x_offline[:,0],phi,M,R)
# t_off_prev = time_offline[0]
#
# ###---Start the loop here:
# while (np.linalg.norm(dtheta)> tol):
#
#     #Time delay need otherwise execution happens in same time instance
#
#     # dtheta = np.copy(theta)
#     x_off = x_offline[:,N]
#     t_off = time_offline[N]
#
#     u_off = compute_u(Ke,x_off,t_off)
#     d_off = cost_func(x_off,u_off,M,R)
#     d_policy_off = cost_func(x_off,phi,M,R)
#     ddf = (d_policy_off-d_policy_off_prev) / (t_off - t_off_prev)
#     d_policy_off_prev = d_policy_off
#     t_off_prev = t_off
#
#
#
#     c = cost_func(x_off,u_off,M,R)
#     d = cost_func(x_off,u_off,M,R)
#     x = x.squeeze()
#     U1 = np.concatenate((x_off,u_off))  # U with u
#     U2 = np.concatenate((x_off,phi))  # U with phi
#     Si = kronecker(U1, U1, n, m)
#
#     # Implement eq 23 first:
#     Q = Q_func(d, Si, theta)
#     Q_all.append(Q)
#
#     #Implement eq 24:
#
#     #Define zeta:
#
#     zeta_pt1 = kronecker(U2, U2, n, m)
#     zeta_pt2 = kronecker(U1, U1, n, m)
#     zeta_pt3 = diff_Si(U2, n, m) #(zeta_pt1 - zeta_pt1_prev) / (t - t_prev)
#     zeta_pt3 = zeta_pt3.reshape([np.size(zeta_pt1), 1])
#     zeta = zeta_pt1 - zeta_pt2 + zeta_pt3
#     #zeta_pt1_prev = zeta_pt1
#     zeta_record.append(zeta)
#
#     #Define b(t):
#     t = time.time() - start
#     time_record.append(t)
#
#     dphi = cost_func(x,phi,M,R)
#     dphi = np.atleast_1d(dphi).reshape(1,1)
#     #ddf = (d-d_prev)/(t - t_prev)#diff_d(x,phi,M,R).reshape(2,1)
#     #print("d",d)
#     #print("d_prev",d_prev)
#     #print("t",t)
#     #print("t_prev",t_prev)
#     d_prev = d
#     t_prev = t
#     ddf = np.atleast_1d(ddf)
#     b = c - d + d_policy_off + ddf #diff_d(x,phi,M,R)
#     print("c",c)
#     print("d",d)
#     print("dpoloff",d_policy_off)
#     print("ddf",ddf)
#
#     b = b.squeeze()
#     b = np.atleast_1d(b)
#     print("b",b)
#
#     a = g/(t+1)
#     #if (N<1):
#     #    G_hat_inv = np.ones((s,s))
#     #else:
#     #    G_hat_inv = G_inv_func(time_record, zeta_record, s)
#     dtheta = diff_theta(zeta, b, a, theta, np.eye(s))
#     # print(dtheta)
#     print("zeta", zeta)
#     #print("b", b)
#     #print("a", a)
#     print("theta before update", theta)
#     print("dtheta before update", dtheta)
#     #Update theta
#     theta = theta - dtheta
#     theta_record.append(theta)
#
#     #Implement eq 22:
#
#     # phi = np.argmin(Q, axis = 1)
#     # print("Arg is: ",np.argmin(Q_all, axis = 1))
#     # phi = Q_all[np.argmin(Q_all, axis = 1).squeeze()]
#     # phi = phi.squeeze()
#     dim1 = np.int(n * (n + 1) / 2 + 1)
#     dim2 = np.int(n * (n + 1) / 2 + n * m)
#     K_N = np.array((theta[dim1], theta[dim2])).T / theta[-m]  # Quu^-1*Qxu
#     phi = np.matmul(K_N, x.reshape(2, 1))
#     phi = phi.squeeze()
#     phi = np.atleast_1d(phi)
#
#     #phi = Q.squeeze()
#     #phi = np.atleast_1d(phi)
#     #phi_record.append(phi)
#
#
#     u = compute_u(Ke, x, t)
#     x = np.matmul(A, x).reshape(2,1) + np.matmul(B.reshape(2,1), u.reshape(1,1))
#     print("N:",N)
#     #print("x:",x)
#     #print("Si", Si)
#     print("theta",theta)
#     #print("phi",phi)
#     #print("u",u)
#
#     N += 1
#     print("observe", np.linalg.norm(dtheta))
#
#
#
# ##====End of while loop
#
# #Plot functions:
# plt.plot(theta_record, t)
# plt.title("Plot of theta")
# plt.xlabel("time (t)")
# plt.ylabel("theta")
# plt.show()


#=======================================================================================================================
##This document implements the QSA off policy iteration
import numpy as np
# from tools import kronecker
import time
import matplotlib.pyplot as plt
import pandas as pd
#Helper functions:

def readfile_func(filename):
    '''This function gets the offline values of time, states and input from given csv file ---> filename'''
    datafile = pd.read_csv('{}.csv'.format(filename))
    print(datafile.head())
    t = datafile['t'].to_numpy()
    x1 = datafile['x1'].to_numpy()
    x2 = datafile['x2'].to_numpy()
    u = datafile['u'].to_numpy()
    x = np.concatenate((x1,x2), axis=0)
    n = x1.shape[0]
    x = x.reshape(2, n)
    return t, x, u

# def kronecker(A,B,n,m):
#
#     k = A.shape[0]
#     s = (int(1/2*((k)*(k+1))),1)
#     C = np.ones(s)
#
#     for i in range(n):
#         C[i]=A[i]*B[i]
#
#     for i in range(n+m-1):
#         for j in range(i,n+m-1):
#             C[i+n]=A[i]*B[j]
#
#     for i in range(1,m+1):
#         C[-i]=A[-i]*B[-i]
#
#     return C
def kronecker(A, B, n, m):
    # A=np.array([[1],[2],[3],[4]])
    # B=np.array([[1],[2],[3],[4]])
    k = A.shape[0]
    # print(k)
    # [1*1 , 2*2, 1*2, 1*3,2*3,3*3]
    # [x1**2, x2**2, x1x2, x1u, x2u, u**2]
    # [x1**2, x2**2, x1x2, x1u1, x1u2, x2u1,x2u2, u1u2 ,u1**2, u2**2]

    s = int(1 / 2 * ((n + m) * (n + m + 1)))
    C = np.zeros(s)

    for i in range(n):
        C[i] = A[i] * B[i]

    c = n
    for i in range(n + m - 1):
        for j in range(i + 1, n + m):
            C[c] = A[i] * B[j]
            c += 1

    for i in range(1, m + 1):
        C[-i] = A[-i] * B[-i]

    return C.reshape(s, 1)

def diff_Si(U,Ke,xdot, n, m):
    # U: si in paper e.g
    # n
    s = 0.5*(len(U)*(len(U)+1))
    s = np.int(s)
    d = np.zeros(s)

    for i in range(n):
        d[i] = 2*U[i]*xdot[i]

    #for i in range(n + m - 1):
    #    for j in range(i, n + m - 1):
    #        d[i + n] = U[i]*(np.matmul(Ke,xdot)) + U[j]

    d[n]=xdot[0]*U[1]+U[0]*xdot[1]

    for i in range(2):
        d[i+n+m]=xdot[i]*np.matmul(Ke,U[0:2])+U[i]*np.matmul(Ke,xdot) #add 'epsilon' later

    for i in range(1, m + 1):
        d[-i] = 2*U[-i]*(np.matmul(Ke,xdot)) #add "epsilon" later

    return d

def diff_d(x,u,xdot,udot,M,R):

    diff_d = 2*np.matmul(np.matmul(x.T,M),xdot) + 2* np.matmul(np.matmul(u.T,R),udot)
    return diff_d

def cost_func(x,u,M,R):
    cost = np.matmul(x.T,np.matmul(M,x)) + np.matmul(u.T,np.matmul(R,u))
    return cost

def Q_func(d, Si, theta):
    Q = d + np.matmul(theta.T, Si)

    return Q

def eps_func(Q, Qscore, c, zeta, b, a, theta, M, R, x, phi, n, m):
    dQscore = diff_d(x, phi ,M,R) + np.matmul(theta.T,diff_Si(np.concatenate(x,phi), m, n)) + np.matmul(diff_theta(zeta, b, a, theta),kronecker(np.concatenate(x,phi), np.concatenate(x,phi), n, m))
    eps = -Q + Qscore + c + dQscore

    return eps

def diff_theta(zeta, b, a, theta, G_hat_inv):
    dtheta_pt1 = (np.matmul(zeta.T, theta) + b)
    dtheta = -a * np.matmul(G_hat_inv, dtheta_pt1*zeta)
    return dtheta

def compute_u(Ke, x, t):
    q = 24 #Choosen in the paper as number of sinusoids
    zeta_t = 0
    freq = np.random.choice(np.linspace(0, 1000), q)    # Frequency (0-50 rad/s)
    phase = np.random.choice(np.linspace(0, 2*np.pi), q)    # Phase (0-1 rad)
    a = np.random.rand(q,1)/1000                           # Amplitude (0-0.01)
    for i in range(q):      # Compute sinusoidal noise to be introduced with input
        zeta_t += a[i]*np.sin(freq[i]*t + phase[i])

    u = np.matmul(Ke,x) + zeta_t
    # u = u.reshape(1,1)#np.atleast_1d(u)

    return u

def G_inv_func(time_record, zeta_record, s):
    integrand = np.zeros((s, s))
    for i in range(1,len(time_record)-1):
        integrand += (time_record[i] - time_record[i-1]) * (np.matmul(np.array(zeta_record[i-1:i]).reshape(s,1),np.array(zeta_record[i-1:i]).reshape(1,s)) + np.matmul(np.array(zeta_record[i:i+1]).reshape(s,1),np.array(zeta_record[i:i+1]).reshape(1,s))) / 2
    G_hat = integrand / time_record[-1]
    return np.linalg.pinv(G_hat)

def double_integrator_with_friction_noise(t, x, K):
    x1, x2 = x
    u = compute_u(K,x,t)


    x_1_dot = x2#-x2
    x_2_dot = -0.1 * x2 + u

    return [x_1_dot, x_2_dot]
#####============STRUCTURE OF CODE==============
""""
1. Initialize all variables outside
2. While loop till theta - dtheta converges
3. counter of N = N + 1
"""
# run simulation
#from dynamic_system_simulation import cart_pendulum_sim_lqr
from scipy import integrate
filename = "offlinedata"
[time_offline, x_offline, u_offline] = readfile_func(filename)
print(x_offline[1:].shape)
print(time_offline.reshape(1,500).shape)

plt.plot(x_offline[1,:], label = 'x2')
plt.plot(x_offline[0,:], label = 'x1')
plt.plot(u_offline, label = 'u')
plt.legend()
plt.title("Offline trajectories")
plt.show()

M = 0.5  # cart mass
m = 0.2  # pendulum mass
g = 9.81  # gravity
L = 1  # pendulum length
f = 0.1  # friction
b= 0.5 #friction for cart
F = 1  # control input [N]
Ke = np.array([-0.3162 ,   0.7617])
K_N=np.array([0,0])

#Initialize
##Define known case for testing
A = np.array([[0, -1],[0, -0.1]])
B = np.array([0,1])
M = np.eye(2)
R = 10*np.eye(1)

g = 1.5  #Authors experimented with different values

N = 1  #Defines number of iterations

#Initialize K #In paper page 5, K = [5, 1]
K = np.array([-1, 0])
Ke = np.array([-1, -2])

#Initialize states, input and parameters
# Start the simulation time (we can look for better place)
start = time.time()
t = 0
t_prev = 0

#Define initial states and inputs
x = np.array([0.0, 0.0])
u = compute_u(Ke, x, t)



U = np.concatenate((x, u), axis=0)

s = 0.5*(len(U)*(len(U)+1))     # Size of kronecker array
s = np.int(s)

n = np.size(x)  #Sates size
m = np.size(u)  #Input size

phi = np.matmul(K,x)
phi = np.atleast_1d(phi)

size_theta = np.size(kronecker(U, U, m, n))   #Just used for gettin size for theta

theta = np.zeros(size_theta)
theta = theta.reshape(size_theta,1)
dtheta = np.ones(size_theta)
dtheta = dtheta.reshape(size_theta,1)
d_prev = cost_func(x,u,M,R) - 10
zeta_pt1_prev = kronecker(U, U, n, m)

theta_record = []
phi_record = []
time_record = []
Q_all = []
zeta_record = []

theta_record.append(theta)
time_record.append(t)
tol = 1e-3

d_policy_off_prev = cost_func(x_offline[:,0],phi,M,R)
t_off_prev = time_offline[0]

print('========================Start of while loop========================')

###---Start the loop here:
while (np.linalg.norm(dtheta)> tol):

    #Time delay need otherwise execution happens in same time instance

    # dtheta = np.copy(theta)
    x_off = x_offline[:,N]
    t_off = time_offline[N]

    u_off = compute_u(Ke,x_off,t_off)
    d_off = cost_func(x_off,u_off,M,R)
    # print("x_off", x_off)
    # print("u_off", u_off)
    d_policy_off = cost_func(x_off,phi,M,R)

    t_off_prev = t_off

    c = cost_func(x_off,u_off,M,R)
    d = cost_func(x_off,u_off,M,R)
    x = x.squeeze()
    U1 = np.concatenate((x_off,u_off))  # U with u
    U2 = np.concatenate((x_off,phi))  # U with phi
    Si = kronecker(U1, U1, n, m)

    # Implement eq 23 first:
    Q = Q_func(d, Si, theta)
    Q_all.append(Q)

    #Implement eq 24:

    #Define zeta:

    zeta_pt1 = kronecker(U2, U2, n, m)
    zeta_pt2 = kronecker(U1, U1, n, m)
    xdot=np.array(double_integrator_with_friction_noise(t_off, x_off, Ke)).T
    zeta_pt3 = diff_Si(U2,Ke,xdot, n, m) #(zeta_pt1 - zeta_pt1_prev) / (t - t_prev)
    zeta_pt3 = zeta_pt3.reshape([np.size(zeta_pt1), 1])
    zeta = zeta_pt1 - zeta_pt2 + zeta_pt3
    #zeta_pt1_prev = zeta_pt1
    zeta_record.append(zeta)

    #zeta_pt1_prev = zeta_pt1
    zeta_record.append(zeta)

    #Define b(t):
    t = time.time() - start
    time_record.append(t)

    dphi = cost_func(x,phi,M,R)
    dphi = np.atleast_1d(dphi).reshape(1,1)
    phidot=np.matmul(K_N,xdot)

    ddf=diff_d(x_off, u_off, xdot, phidot, M, R)

    d_prev = d
    t_prev = t
    ddf = np.atleast_1d(ddf)
    b = c - d + d_policy_off + ddf #diff_d(x,phi,M,R)
    # print("c",c)
    # print("d",d)
    # print("dpoloff",d_policy_off)
    # print("ddf",ddf)

    b = b.squeeze()
    b = np.atleast_1d(b)
    # print("b",b)

    a = g/(t+1)

    #if (N<1):
    #    G_hat_inv = np.ones((s,s))
    #else:
    #    G_hat_inv = G_inv_func(time_record, zeta_record, s)

    dtheta = diff_theta(zeta, b, a, theta, np.eye(s))

    # print("zeta", zeta)

    print("theta before update", theta)
    print("dtheta before update", dtheta)
    #Update theta
    theta = theta - dtheta
    theta_record.append(theta)

    #Implement eq 22:

    dim1 = np.int(n * (n + 1) / 2 + 1)
    dim2 = np.int(n * (n + 1) / 2 + n * m)
    K_N = np.array((theta[dim1], theta[dim2])).T / theta[-m]  # Quu^-1*Qxu
    phi = np.matmul(K_N, x.reshape(2, 1))
    phi = phi.squeeze()
    phi = np.atleast_1d(phi)
    print("phi: ", phi)


    u = compute_u(Ke, x, t)
    x = np.matmul(A, x).reshape(2,1) + np.matmul(B.reshape(2,1), u.reshape(1,1))
    print("N:",N)
    print("theta after update: ",theta)

    N += 1



##====End of while loop

#Plot functions:

plt.plot(theta_record, t)
plt.title("Plot of theta")
plt.xlabel("time (t)")
plt.ylabel("theta")
plt.show()
