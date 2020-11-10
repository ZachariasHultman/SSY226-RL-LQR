import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt



def cart_pendulum_sim(t , x, L=1., m=1., M = 1., g=9.81, F=0, f=0,f_cart=0):
    """
    x_1_dot = velocity
    x_2_dot = acceleration
    x_3_dot = angular velocity
    x_4_dot = angular acceleration
    """
    # print(x)
    x1, x2, x3, x4 = x
    x_2_dot_nomi = (-m*g*np.sin(x3)*np.cos(x3) +
                    m*L*x4*x4*np.sin(x3) +
                    f*m*x4*np.cos(x3)+F)-x2*f_cart

    x_2_dot_denomi = (M + (1 - np.cos(x3) * np.cos(x3)) * m)

    x_4_dot_nomi = ((M+m)*(g*np.sin(x3)-f*x4) -
                    (L*m*x4*x4*np.sin(x3)+F) * np.cos(x3))

    x_4_dot_denomi = L*x_2_dot_denomi

    x_1_dot = x2
    x_2_dot = x_2_dot_nomi/x_2_dot_denomi
    x_3_dot = x4
    x_4_dot = x_4_dot_nomi/x_4_dot_denomi ###

    return [x_1_dot, x_2_dot, x_3_dot, x_4_dot]
