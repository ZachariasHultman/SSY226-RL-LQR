import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


def cart_pendulum_sim(theta_init, x_init, t, L=1, m=1, b=0, g=9.81, F=0, f=0):

    x_dot_2_nomi = (-m*g*np.sin(theta_init[0])*np.cos(theta_init[0]) +
                    m*l*theta_init[1]*theta_init[1]*np.sin(theta_init[0]) +
                    f*m*theta_init[1]*np.cos(theta_init[0])+F)

    x_dot_2_denomi = (M + (1 - np.cos(theta_init[0]) * np.cos(theta_init[0])) * m)

    theta_dot_2_nomi = (M+m)*(g*np.sin(theta_init[0])-f*theta_init[1]) - \
        (l*m*theta_init[1]*theta_init[1]*np.sin(theta_init[0])+F) * \
        np.cos(theta_init[0])

    theta_dot_2_denomi = l*x_dot_2_denomi

    x_dot_1 = x_init[1]
    x_dot_2 = x_dot_2_nomi/x_dot_2_denomi
    theta_dot_1 = theta_init[1]
    theta_dot_2 = theta_dot_2_nomi/theta_dot_2_denomi

    return x_dot_1, x_dot_2, theta_dot_1, theta_dot_2

M = .5
m = 0.2
b = 0.1
I = 0.006
g = 9.8
l = 0.3


vals = integrate.odeint(cart_pendulum_sim, theta_init, t)

