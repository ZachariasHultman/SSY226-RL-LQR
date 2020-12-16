import numpy as np
from scipy import interpolate
from tools import kronecker
from tools import vech_to_mat_sym, sigma_fun, mat_to_vec_sym
# equation 20 is the following function


def approx_update(x_hist,u_hist, W_c_hat, alpha_c, M, R, dt, n ,m,int_term,t,T):

    # print(x_hist[:,-1])
    # print(u_hist)
    U = np.concatenate((x_hist[:,-1].reshape(n,1).T, u_hist[:,-1].reshape(m,1).T),1).T
    if t[-1]<T:
        x_interp=x_hist[:,-1].reshape(n,1)
        u_interp=u_hist[:,-1].reshape(m,1)*0
    else:
        # print('kuken')
        x_tmp=[]
        # print(np.asarray(t).shape)
        # print(np.ravel(u_hist[0]).shape)
        for dim in range(x_hist.shape[0]):
            x_tmp.append(np.interp(t[-1]-T,np.asarray(t),np.ravel(x_hist[dim])))
        x_interp=np.asarray(x_tmp).reshape(n,1)
        u_tmp=[]
        for dim in range(u_hist.shape[0]):
            u_tmp.append(np.interp(t[-1]-T,np.asarray(t),np.ravel(u_hist[dim])))
        u_interp=np.asarray(u_tmp).reshape(m,1)



    # U_prev = np.concatenate((x_hist[:,0].reshape(n,1).T, u_hist[:,0].reshape(m,1).T),1).T
    # print('old',U_prev)

    U_prev = np.concatenate((x_interp.reshape(n,1).T, u_interp.reshape(m,1).T),1).T
    # print(U_prev)
    # br
    # U_prev = np.concatenate((np.trapz(x_hist,dx=dt).reshape(n,1).T, np.trapz(u_hist,dx=dt).reshape(m,1).T),1).T

    n=x_hist[:,-1].shape[0]
    m=u_hist[:,-1].shape[0]
    s = int(1 / 2 * ((n + m) * (n + m + 1)))

    row,col=np.tril_indices(n+m)

    # print(np.reshape(np.kron(U,U),(n+m,n+m)))
    
    U_kron=np.asarray([np.reshape(np.kron(U,U),(n+m,n+m))[c,r] for r,c in zip(col,row)])
    U_prev_kron=np.asarray([np.reshape(np.kron(U_prev,U_prev),(n+m,n+m))[c,r] for r,c in zip(col,row)])

    Q=np.zeros((n+m,n+m))
    count=0
    for r,c in zip(col,row):
        if r==c:
            Q[c,r]=W_c_hat[count]*0.5
        else:
            Q[c,r]=W_c_hat[count]
        count+=1
    Q=Q+Q.T

    # sigma_pt_1= mat_to_vec_sym(np.tril(vech_to_mat_sym(np.kron(U,U),n,m)),n,m) 
    # sigma_pt_2= mat_to_vec_sym(np.tril(vech_to_mat_sym(np.kron(U_prev,U_prev),n,m)),n,m)
    sigma=  U_kron - U_prev_kron
    sigma=sigma.reshape(s,1)
    # Compute new Weights
    # See equation below (e=...) to understand the need of the integral term.
    # The integral term is calculated by assumptions that self-defined matrices M and R are diagonal
    # and assumption that the integration is discrete with time step T and only two points of evaluation

    # int_term=np.zeros(x_hist.shape[1])
    # for k in range(x_hist.shape[1]):
    #     int_term[k] = x_hist[:,k].reshape(n,1).T @ M @ x_hist[:,k].reshape(n,1) + (u_hist[:,k].reshape(m,1).T @ R @ u_hist[:,k].reshape(m,1))

    # int_term=np.trapz(int_term,dx=dt)

    # print(U.T@Q@U )
    # print(U_prev.T@Q@U_prev)
    # print(int_term)

    # Using integral RL gives error of (Bellman) value function as (eq.17 to eq.18). 
    # e = 0.5* ( W_c_hat.T @ U_kron  + int_term - W_c_hat.T @ U_prev_kron)
    e=0.5*( U.T@Q@U - U_prev.T@Q@U_prev +  int_term)  
    
    # print('sigma',sigma)
    # br

    # Update of the critic approximation weights (Equation 20)

    # print((1 + np.matmul(sigma.T, sigma))**2)
    W_c_hat_dot = -alpha_c * sigma / ((1 + sigma.T @ sigma)**2) * e.T
    # W_c_hat_dot = -alpha_c * sigma * e.T

    # print('W_c_hat_dot',W_c_hat_dot.shape)

    # br
    # W_c_tilde_dot = -alpha_c * np.matmul((np.matmul(sigma, sigma.T) / ((1 + np.matmul(sigma.T, sigma))**2)), W_c_tilde)
    # Q_bar_tilde = vech_to_mat_sym(W_c_tilde, n + m)
    # Q_xu_tilde = Q_bar_tilde[n:,:n].T

    return np.ravel(W_c_hat_dot) #, W_c_tilde_dot, Q_xu_tilde


