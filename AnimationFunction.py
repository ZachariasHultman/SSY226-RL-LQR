import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animationfunction(vals, L):
    """
    vals = Class containing all the states and time stamps
    L = Length of pendulum
    """

    ###Animation
    plt.rcParams['animation.html'] = 'html5'

    #####################################################################
    #################=====Create test data=====####################################
    # x2 = np.linspace(1,10,100)
    # y2 = 0*x2
    # angle = np.linspace(0,180,100)
    # angle = np.deg2rad(angle)
    # x1 = x2 + L*np.cos(angle)
    # y1 = y2 + L*np.sin(angle)
    # angle_disp = np.rad2deg(angle)
    # print("The adjusted angle is: ",angle)
    #####################################################################

    #################=====Get actual data=====####################################
    # Get cart position
    x2 = vals.y[:1].T
    y2 = np.abs(0 * vals.y[:1].T)  # Y position of cart always zero

    # Compute pole position

    angle = -vals.y[2:3].T
    x1 = x2 + L * np.cos(angle + np.pi / 2)
    y1 = y2 + L * np.sin(angle + np.pi / 2)
    angle_disp = -np.rad2deg(angle)
    print("Size y1 is:", np.max(x2))

    ##Plot positions
    # plt.figure(3)
    # plt.plot(vals.t, x1, label='m1x')
    # plt.plot(vals.t, x2, label='m2x')
    # plt.plot(vals.t, y1, label='m1y')
    # plt.plot(vals.t, y2, label='m2y')
    # plt.legend(loc="upper left")
    # plt.title("Cart and pendulum position")
    #
    # print("max time is:", np.max(vals.t))

    #Plot animation

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, autoscale_on=False, \
                         xlim=(np.min(x2) - 2, np.max(x2) + 1), ylim=(-1, np.max(y2) + 1))
    ax.set_xlabel('position')
    ax.get_yaxis().set_visible(False)

    crane_rail, = ax.plot([np.min(x2) - 2, np.max(x2) + 1], [-0.05, -0.05], 'k-', lw=4)
    start, = ax.plot([0, 0], [-1.3, 0.5], 'k:', lw=2)
    objective, = ax.plot([np.max(x2), np.max(x2)], [-1.3, 0.5], 'k:', lw=2)
    mass1, = ax.plot([], [], linestyle='None', marker='o', \
                     markersize=10, markeredgecolor='k', \
                     color='orange', markeredgewidth=2)
    mass2, = ax.plot([], [], linestyle='None', marker='s', \
                     markersize=20, markeredgecolor='k', \
                     color='orange', markeredgewidth=2)
    line, = ax.plot([], [], 'o-', color='orange', lw=4, \
                    markersize=10, markeredgecolor='k', \
                    markerfacecolor='k')
    time_template = 'time = %.1fs'
    angle_template = 'angle = %.1fdegree'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    angle_text = ax.text(0.5, 0.9, '', transform=ax.transAxes)
    start_text = ax.text(-0.1, -0.1, 'start', ha='right')
    end_text = ax.text(3.1, -0.1, 'objective', ha='left')

    def init():
        mass1.set_data([], [])
        mass2.set_data([], [])
        line.set_data([], [])
        time_text.set_text('')
        angle_text.set_text('')
        return line, mass1, mass2, time_text, angle_text

    def animate(i):
        mass1.set_data([x1[i]], [y1[i]])
        mass2.set_data([x2[i]], [y2[i]])
        line.set_data([x2[i], x1[i]], [y2[i], y1[i]])
        time_text.set_text(time_template % vals.t[i])
        angle_text.set_text(angle_template % angle_disp[i])
        return line, mass1, mass2, time_text, angle_text

    ani_a = animation.FuncAnimation(fig, animate, \
                                    np.arange(1, len(vals.t)), \
                                    interval=5, blit=False, init_func=init)

    # # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=400, interval=20, blit=True)

    # requires ffmpeg to save mp4 file - Takes too long and creates heavier files
    #  available from https://ffmpeg.zeranoe.com/builds/
    #  add ffmpeg.exe to path such as C:\ffmpeg\bin\ in
    #  environment variables

    # ani_a.save('Pendulum_Control.gif',fps=40)

    print("Angle is: ", angle)

    plt.show()