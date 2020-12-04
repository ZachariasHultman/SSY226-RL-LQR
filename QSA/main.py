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

def diff_Si(U,Ke,xdot,eps,epsdot,n,m):
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
        d[i+n+m]=xdot[i]*np.matmul(Ke,U[0:2])+U[i]*np.matmul(Ke,xdot) + xdot[i]*eps + U[i]*epsdot

    for i in range(1, m + 1):
        d[-i] = 2*np.matmul(Ke,U[0:2])*np.matmul(Ke,xdot) + 2*np.matmul(Ke,xdot)*eps + 2*np.matmul(Ke,U[0:2])*epsdot + 2*eps*epsdot

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

#def eps_func(Q, Qscore, c, zeta, b, a, theta, M, R, x, phi, n, m):
#    dQscore = diff_d(x, phi ,M,R) + np.matmul(theta.T,diff_Si(np.concatenate(x,phi), m, n)) + np.matmul(diff_theta(zeta, b, a, theta),kronecker(np.concatenate(x,phi), np.concatenate(x,phi), n, m))
#    eps = -Q + Qscore + c + dQscore
#
#    return eps

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

def double_integrator_with_friction_noise(t, x, u, K):
    x1, x2 = x
    #u = compute_u(K,x,t)


    x_1_dot = x2#-x2
    x_2_dot = -0.1 * x2 + u[0]

    return [x_1_dot, x_2_dot]
#####============STRUCTURE OF CODE==============
""""
1. Initialize all variables outside
2. While loop till theta - dtheta converges
3. counter of N = N + 1
"""


def readfile_func(filename):
    '''This function gets the offline values of time, states and input from given csv file ---> filename'''
    datafile = pd.read_csv('{}.csv'.format(filename))
    print(datafile.head())
    t = datafile['t'].to_numpy()
    x1 = datafile['x1'].to_numpy()
    x2 = datafile['x2'].to_numpy()
    u = datafile['u'].to_numpy()
    #================================================
    a1 = datafile['a1'].to_numpy()
    a2 = datafile['a2'].to_numpy()
    a3 = datafile['a3'].to_numpy()
    a4 = datafile['a4'].to_numpy()
    a5 = datafile['a5'].to_numpy()
    a6 = datafile['a6'].to_numpy()
    a7 = datafile['a7'].to_numpy()
    a8 = datafile['a8'].to_numpy()
    a9 = datafile['a9'].to_numpy()
    a10 = datafile['a10'].to_numpy()
    a11 = datafile['a11'].to_numpy()
    a12 = datafile['a12'].to_numpy()
    a13 = datafile['a13'].to_numpy()
    a14 = datafile['a14'].to_numpy()
    a15 = datafile['a15'].to_numpy()
    a16 = datafile['a16'].to_numpy()
    a17 = datafile['a17'].to_numpy()
    a18 = datafile['a18'].to_numpy()
    a19 = datafile['a19'].to_numpy()
    a20 = datafile['a20'].to_numpy()
    a21 = datafile['a21'].to_numpy()
    a22 = datafile['a22'].to_numpy()
    a23 = datafile['a23'].to_numpy()
    a24 = datafile['a24'].to_numpy()
    #================================================
    f1 = datafile['f1'].to_numpy()
    f2 = datafile['f2'].to_numpy()
    f3 = datafile['f3'].to_numpy()
    f4 = datafile['f4'].to_numpy()
    f5 = datafile['f5'].to_numpy()
    f6 = datafile['f6'].to_numpy()
    f7 = datafile['f7'].to_numpy()
    f8 = datafile['f8'].to_numpy()
    f9 = datafile['f9'].to_numpy()
    f10 = datafile['f10'].to_numpy()
    f11 = datafile['f11'].to_numpy()
    f12 = datafile['f12'].to_numpy()
    f13 = datafile['f13'].to_numpy()
    f14 = datafile['f14'].to_numpy()
    f15 = datafile['f15'].to_numpy()
    f16 = datafile['f16'].to_numpy()
    f17 = datafile['f17'].to_numpy()
    f18 = datafile['f18'].to_numpy()
    f19 = datafile['f19'].to_numpy()
    f20 = datafile['f20'].to_numpy()
    f21 = datafile['f21'].to_numpy()
    f22 = datafile['f22'].to_numpy()
    f23 = datafile['f23'].to_numpy()
    f24 = datafile['f24'].to_numpy()
    #================================================
    p1 = datafile['p1'].to_numpy()
    p2 = datafile['p2'].to_numpy()
    p3 = datafile['p3'].to_numpy()
    p4 = datafile['p4'].to_numpy()
    p5 = datafile['p5'].to_numpy()
    p6 = datafile['p6'].to_numpy()
    p7 = datafile['p7'].to_numpy()
    p8 = datafile['p8'].to_numpy()
    p9 = datafile['p9'].to_numpy()
    p10 = datafile['p10'].to_numpy()
    p11 = datafile['p11'].to_numpy()
    p12 = datafile['p12'].to_numpy()
    p13 = datafile['p13'].to_numpy()
    p14 = datafile['p14'].to_numpy()
    p15 = datafile['p15'].to_numpy()
    p16 = datafile['p16'].to_numpy()
    p17 = datafile['p17'].to_numpy()
    p18 = datafile['p18'].to_numpy()
    p19 = datafile['p19'].to_numpy()
    p20 = datafile['p20'].to_numpy()
    p21 = datafile['p21'].to_numpy()
    p22 = datafile['p22'].to_numpy()
    p23 = datafile['p23'].to_numpy()
    p24 = datafile['p24'].to_numpy()
    x = np.concatenate((x1,x2), axis=0)
    a = np.concatenate((a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24), axis=0)
    f = np.concatenate((f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24), axis=0)
    p = np.concatenate((p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24), axis=0)
    n = x1.shape[0]
    x = x.reshape(2, n)
    a = a.reshape(24, n)
    f = f.reshape(24, n)
    p = p.reshape(24, n)
    return t, x, u, a, f, p

def eps_func(t, a, f, p):
    e = 0
    for i in range(24):
        e += a[i]*f[i]*np.sin(f[i]*t + p[i])
    return e

def diff_eps(t, a, f, p):
    dedt = 0
    for i in range(24):
        dedt += a[i]*f[i]*np.cos(f[i]*t + p[i])
    return dedt


def vech_to_mat_sym(a, n, m):
    """
    Takes a vector which represents the elements in a upper (or lower)
    triangular matrix and returns a symmetric (n x n) matrix.
    The off-diagonal elements are divided by 2.
    :param a: input vector of type np array and size n(n+1)/2
    :param n: dimension of symmetric output matrix A
    :return: symmetric matrix of type np array size (n x n)
    """
    s = int(1 / 2 * ((n + m) * (n + m + 1)))
    A = np.ndarray((n + m, n + m))

    for tmp in range(n):
        A[tmp, tmp] = a[tmp]

    c = m
    for j in range(n, n + m):
        for i in range(j, n + m):
            for tmp in range(s - c, s):
                A[i, i] = a[tmp]
                c -= 1
                break

    c = n
    for j in range(n + m):
        for i in range(j + 1, n + m):
            A[i, j] = a[c] / 2
            A[j, i] = a[c] / 2
            c += 1
    return A


def mat_to_vec_sym(A, n, m):
    s = int(1 / 2 * ((n + m) * (n + m + 1)))
    a = np.ndarray((s))
    c = 0
    # [1,4,3,5,6,7,8,11,9,12]

    for tmp in range(n):
        a[tmp] = A[tmp, tmp]

    c = m
    for j in range(n, n + m):
        for i in range(j, n + m):
            for tmp in range(s - c, s):
                a[tmp] = A[i, i]
                c -= 1
                break

    c = n
    for j in range(n + m):
        for i in range(j + 1, n + m):
            a[c] = A[i, j] * 2
            c += 1
    return a

# run simulation
#from dynamic_system_simulation import cart_pendulum_sim_lqr
from scipy import integrate
filename = "offlinedata"
time_offline, x_offline, u_offline, amp, fr, p = readfile_func(filename)
#[time_offline, x_offline, u_offline] = readfile_func(filename)
#print(x_offline[1:].shape)
#print(time_offline.reshape(1,500).shape)

#plt.plot(x_offline[1,:], label = 'x2')
#plt.plot(x_offline[0,:], label = 'x1')
#plt.plot(u_offline, label = 'u')
#plt.legend()
#plt.title("Offline trajectories")
#plt.show()

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

    u_off = np.array([u_offline[N]])
    #d_off = cost_func(x_off,u_off,M,R)
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
    xdot=np.array(double_integrator_with_friction_noise(t_off, x_off, u_off, Ke)).T
    eps = eps_func(t_off, amp[:,N], fr[:,N], p[:,N])
    epsdot = diff_eps(t, amp[:,N], fr[:,N], p[:,N])
    zeta_pt3 = diff_Si(U2,Ke,xdot,eps,epsdot,n,m) #(zeta_pt1 - zeta_pt1_prev) / (t - t_prev)
    zeta_pt3 = zeta_pt3.reshape([np.size(zeta_pt1), 1])
    zeta = zeta_pt1 - zeta_pt2 + zeta_pt3
    #zeta_pt1_prev = zeta_pt1
    zeta_record.append(zeta)

    #zeta_pt1_prev = zeta_pt1
    zeta_record.append(zeta)

    #Define b(t):
    t = time.time() - start
    time_record.append(t_off)

    dphi = cost_func(x,phi,M,R)
    dphi = np.atleast_1d(dphi).reshape(1,1)
    phidot=np.array([np.matmul(K_N,xdot)])

    #changed phidot to array
    ddf=diff_d(x_off, u_off, xdot, phidot, M, R)

    d_prev = d
    t_prev = t
    ddf = np.atleast_1d(ddf)
    b = c - d + d_policy_off + ddf #diff_d(x,phi,M,R)
    #print("c",c)
    #print("d",d)
    #print("d_policy",d_policy_off)
    #print("ddt d_policy",ddf)

    b = b.squeeze()
    b = np.atleast_1d(b)


    a = g/(t+1)

    #if (N<1):
    #    G_hat_inv = np.ones((s,s))
    #else:
    #    G_hat_inv = G_inv_func(time_record, zeta_record, s)

    dtheta = diff_theta(zeta, b, a, theta, np.eye(s))

    print("theta before update", theta)
    #print("zeta", zeta)
    #print("b", b)
    #print("a", a)
    print("dtheta", dtheta)
    #Update theta
    factor=1
    theta = theta - factor*dtheta
    theta_record.append(theta)

    #Implement eq 22:

    #dim1 = np.int(n * (n + 1) / 2 + 1)
    #dim2 = np.int(n * (n + 1) / 2 + n * m)
    #K_N = np.array((theta[dim1], theta[dim2])).T / theta[-m]  # Quu^-1*Qxu
    #print("K_N", K_N)
    Qbar = vech_to_mat_sym(theta, n, m)
    Qux = Qbar[-1][0:2]
    K_N = -np.linalg.pinv(R)*Qux
    phi = np.matmul(-np.linalg.pinv(R)*Qux,x_off)
    print("K_N",K_N)
    #phi = np.matmul(K_N, x.reshape(2, 1))
    phi = phi.squeeze()
    phi = np.atleast_1d(phi)
    phi_record.append(phi)
    print("phi: ", phi)


    u = compute_u(Ke, x, t)
    x = np.matmul(A, x).reshape(2,1) + np.matmul(B.reshape(2,1), u.reshape(1,1))
    print("N:",N)
    print("theta after update: ",theta)

    N += 1



##====End of while loop

#Plot functions:

len_time_theta = np.shape(time_record)[0]
print("final time", time_offline[len_time_theta-1])
len_phi = np.shape(phi_record)
thetarect = np.array(theta_record[0:len_time_theta]).reshape(s,len_time_theta)


plt.plot(time_record, thetarect[0,:])
plt.plot(time_record, thetarect[1,:])
plt.plot(time_record, thetarect[2,:])
plt.plot(time_record, thetarect[3,:])
plt.plot(time_record, thetarect[4,:])
plt.plot(time_record, thetarect[5,:])
plt.title("Plot of theta")
plt.xlabel("time (t)")
plt.ylabel("theta")
plt.show()

#timerect = np.array(time_record[0:len_time_theta]).reshape(len_time_theta,1)
#timerect = timerect[0:len_phi-1][0]
#print("shape of ohi", np.shape(timerect))
#rint("tiem 15",time_record[15])
#print("phi 15",phi_record[15])
#phirect = np.array(phi_record[0:len_phi].reshape(len_phi,1))
#plt.plot(timerect, phirect)
#plt.title("Plot of phi")
#plt.xlabel("time (t)")
#plt.ylabel("phi")
#plt.show()
