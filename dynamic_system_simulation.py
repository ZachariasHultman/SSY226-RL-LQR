import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from AnimationFunction import animationfunction

def cart_pendulum_sim(t , x, L=1., m=1., M = 1., g=9.81, F=0, f=0):
    """
    x_1_dot = velocity
    x_2_dot = acceleration
    x_3_dot = angular velocity
    x_4_dot = angular acceleration
    """
    print(x)
    x1, x2, x3, x4 = x
    x_2_dot_nomi = (-m*g*np.sin(x3)*np.cos(x3) +
                    m*L*x4*x4*np.sin(x3) +
                    f*m*x4*np.cos(x3)+F) - x2*f

    x_2_dot_denomi = (M + (1 - np.cos(x3) * np.cos(x3)) * m)

    x_4_dot_nomi = (M+m)*(g*np.sin(x3)-f*x4) - \
        (L*m*x4*x4*np.sin(x3)+F) * \
        np.cos(x3)

    x_4_dot_denomi = L*x_2_dot_denomi

    x_1_dot = x2
    x_2_dot = x_2_dot_nomi/x_2_dot_denomi
    x_3_dot = x4
    x_4_dot = x_4_dot_nomi/x_4_dot_denomi

    return [x_1_dot, x_2_dot, x_3_dot, x_4_dot]


M = 0.5  # cart mass
m = 0.2  # pendulum mass
g = 9.81  # gravity
L = 1 #0.3  # pendulum length
f = 0  # friction
F = 100  # control input [N]

x_init = [0, 0, np.pi, 0]  # Initial state

t_span = [0, 3]  # Time span for simulation

args = (L, m, M, g, F, f)
t_eval = np.linspace(t_span[0],t_span[1],100)

vals = integrate.solve_ivp(cart_pendulum_sim, t_span, x_init, t_eval = t_eval , args=args)

animationfunction(vals, L)

# plt.figure(1)
# plt.subplot(2,2,1)
# plt.plot(vals.t,vals.y[:1].T,label='pos')
# plt.title("position")
# plt.subplot(2,2,2)
# plt.plot(vals.t,vals.y[1:2].T,label='vel')
# plt.title("velocity")
# plt.subplot(2,2,3)
# plt.plot(vals.t,vals.y[2:3].T,label='theta')
# plt.title("angle")
# plt.subplot(2,2,4)
# plt.plot(vals.t,vals.y[3:4].T,label='theta_dot')
# plt.title("angular velocity")
# plt.legend(loc="upper left")
# # plt.show()
# plt.figure(2)
# plt.plot(vals.t,vals.y[:1].T,label='pos')
# plt.plot(vals.t,vals.y[1:2].T,label='vel')
# plt.plot(vals.t,vals.y[2:3].T,label='theta')
# plt.plot(vals.t,vals.y[3:4].T,label='theta_dot')
# plt.legend(loc="upper left")
# # plt.show()
#
# # ##Animation
# plt.rcParams['animation.html'] = 'html5'
#
# #####################################################################
# #################=====Create test data=====####################################
# x2 = np.linspace(1,10,100)
# y2 = 0*x2
# angle = np.linspace(0,180,100)
# angle = np.deg2rad(angle)
# x1 = x2 + L*np.cos(angle)
# y1 = y2 + L*np.sin(angle)
# angle_disp = np.rad2deg(angle)
# print("The adjusted angle is: ",angle)
# #####################################################################
#
#
# #################=====Plot actual data=====####################################
# # # Get cart position
# # x2 = vals.y[:1].T
# # y2 = 0*vals.y[:1].T #Y position of cart always zero
# #
# # #Compute pole position
# # # angle = np.rad2deg(vals.y[2:3].T)
# # angle = vals.y[2:3].T
# # x1 = x2 + L*np.cos(angle-np.pi/2)
# # y1 = y2 + L*np.sin(angle-np.pi/2)
# #angle_disp = np.rad2deg(angle)
#
#
# ##Plot animation
# plt.figure(3)
# plt.plot(x1,y1,label='m1')
# plt.plot(x2,y2,label='m2')
# plt.legend(loc="upper left")
#
# fig = plt.figure(figsize=(10,100))
# ax = fig.add_subplot(111,autoscale_on=False,\
#                      xlim=(-1,np.max(x2)+1),ylim=(-1,np.max(x2)+1))
# ax.set_xlabel('position')
# ax.get_yaxis().set_visible(False)
#
# crane_rail, = ax.plot([-10,np.max(x2)],[-0.3,-0.3],'k-',lw=4)
# start, = ax.plot([0,0],[-1.3,0.5],'k:',lw=2)
# objective, = ax.plot([np.max(x2),np.max(x2)],[-1.3,0.5],'k:',lw=2)
# mass1, = ax.plot([],[],linestyle='None',marker='o',\
#                  markersize=10,markeredgecolor='k',\
#                  color='orange',markeredgewidth=2)
# mass2, = ax.plot([],[],linestyle='None',marker='s',\
#                  markersize=20,markeredgecolor='k',\
#                  color='orange',markeredgewidth=2)
# line, = ax.plot([],[],'o-',color='orange',lw=4,\
#                 markersize=10,markeredgecolor='k',\
#                 markerfacecolor='k')
# time_template = 'time = %.1fs'
# angle_template = 'angle = %.1fdegree'
# time_text = ax.text(0.05,0.9,'',transform=ax.transAxes)
# angle_text = ax.text(0.5,0.9,'',transform=ax.transAxes)
# start_text = ax.text(0,-0.1,'start',ha='right')
# end_text = ax.text(-1+np.max(x2),-0.1,'objective',ha='left')
#
# def init():
#     mass1.set_data([],[])
#     mass2.set_data([],[])
#     line.set_data([],[])
#     time_text.set_text('')
#     angle_text.set_text('')
#     return line, mass1, mass2, time_text, angle_text
#
# def animate(i):
#     mass1.set_data(x1[i],[y1[i]])
#     mass2.set_data([x2[i]],[y2[i]])
#     line.set_data([x2[i],x1[i]],[y2[i],y1[i]])
#     # line.set_data([1,1],[0,0+L])
#     time_text.set_text(time_template % vals.t[i])
#     angle_text.set_text(angle_template % angle_disp[i])
#     return line, mass1, mass2, time_text, angle_text
#
# ani_a = animation.FuncAnimation(fig, animate, \
#          np.arange(1,len(vals.t)), \
#          interval=40,blit=False,init_func=init)
# #
# #
# # # call the animator.  blit=True means only re-draw the parts that have changed.
# anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=200, interval=20, blit=True)
#
# # requires ffmpeg to save mp4 file
# #  available from https://ffmpeg.zeranoe.com/builds/
# #  add ffmpeg.exe to path such as C:\ffmpeg\bin\ in
# #  environment variables
#
# #ani_a.save('Pendulum_Control.mp4',fps=30)
#
# print("Angle is: ",np.rad2deg(angle))
#
# plt.show()