import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


def cart_pendulum_sim(t , x, L=1., m=1., M = 1., g=9.81, F=0, f=0):
    """
    x_1_dot = velocity
    x_2_dot = acceleration
    x_3_dot = angular velocity
    x_4_dot = angular acceleration
    """

    x1, x2, x3, x4 = x
    x_2_dot_nomi = (-m*g*np.sin(x3)*np.cos(x3) +
                    m*L*x4*x4*np.sin(x3) +
                    f*m*x4*np.cos(x3)+F)

    x_2_dot_denomi = (M + (1 - np.cos(x3) * np.cos(x3)) * m)

    x_4_dot_nomi = ((M+m)*(g*np.sin(x3)-f*x4) -
                    (L*m*x4*x4*np.sin(x3)+F) * np.cos(x3))

    x_4_dot_denomi = L*x_2_dot_denomi

    x_1_dot = x2
    x_2_dot = x_2_dot_nomi/x_2_dot_denomi
    x_3_dot = x4
    x_4_dot = x_4_dot_nomi/x_4_dot_denomi ###

    return [x_1_dot, x_2_dot, x_3_dot, x_4_dot]


M = 0.5  # cart mass
m = 0.2  # pendulum mass
g = 9.81  # gravity
L = 0.3  # pendulum length
f = 0  # friction
F = 0  # control input [N]

x_init = [0, 0, np.pi, 0]  # Initial state

t_span = [0, 5]  # Time span for simulation


args = (L, m, M, g, F, f)
t_eval = np.linspace(t_span[0],t_span[1],10)
vals = integrate.solve_ivp(cart_pendulum_sim, t_span, x_init, t_eval=t_eval, args=args)
#x_vals = integrate.odeint(cart_pendulum_sim, theta_init, x_init, t)

print(vals.t)
print(vals.y)
