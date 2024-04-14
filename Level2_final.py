# imports the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv
import numpy as np
import matplotlib.animation as animation

#timestep 
dt = 0.001
t = np.arange(0,10, dt)

#desired trajectory
x = np.cos(t)
y = 2.1+np.sin(t)

#length of links
l1 = 1
l2  = 2

#initialize joint_angles array to store joint angles
#joint_angles[0] stores theta1 and joint_angles[1] stores theta2
#joint_angles[:, i] stores joint angles at ith timestep
#2 rows and len(x) columns
joint_angles = np.zeros((2, len(x)))


#calculating jacobian and end effector position(forward kinematics)
def func(theta1, theta2):
    J = [[-l1*np.sin(theta1) - l2*np.sin(theta1 + theta2), -l2*np.sin(theta1 + theta2)],
     [ l1*np.cos(theta1) + l2*np.cos(theta1 + theta2),  l2*np.cos(theta1 + theta2)]]

    X = l1*np.cos(theta1) + l2*np.cos(theta1+theta2)
    Y = l1*np.sin(theta1) + l2*np.sin(theta1+theta2)
    return J, X, Y

#storing end effector position
Ex = []
Ey=[]

#iterating over all timesteps using inverse kinematics
#E is the error between desired and actual end effector position
#joint_angles[:, i+1] stores joint angles at (i+1)th timestep
#joint angles are updated using pseudo inverse of jacobian
#reference for joint_angles[:, i+1] equation 6.6 in the book Modern RObotics by Kevin Lynch

for i in range(len(x)-1):

    if y[i] <2.9:
        theta1, theta2 = joint_angles[:, i]
        jacobian, x1, y1 = func(theta1, theta2)
        E = np.array([x[i], y[i]]) - np.array([x1, y1])
        Ex.append(x1)
        Ey.append(y1) 
        joint_angles[:, i+1] = joint_angles[:, i] + pinv(jacobian).dot(E)

    else:
        y [i] = 2.9
        theta1, theta2 = joint_angles[:, i]
        jacobian, x1, y1 = func(theta1, theta2)
        E = np.array([x[i], y[i]]) - np.array([x1, y1])
        Ex.append(x1)
        Ey.append(y1) 
        joint_angles[:, i+1] = joint_angles[:, i] + pinv(jacobian).dot(E)

#creates figure and axis
#initalizes an empty plot and end_effector_path
fig = plt.figure()
axis = plt.axes(xlim=(-5,5), ylim=(-5,5))
line, = axis.plot([], [], lw=2)
end_effector_path, = axis.plot(Ex, Ey, '--')


#initializes the plot by setting its data to empty arrays
def init():
    line.set_data([], [])
    return line,

# updates the plot at each animation frame
#calculates position of the two joints at each time step
def animate(i):
    theta1, theta2 = joint_angles[:, i]
    
    x1 = l1*np.cos(theta1)
    y1 = l1*np.sin(theta1)
    x2 = x1 + l2*np.cos(theta2+theta1)
    y2 = y1 + l2*np.sin(theta2+theta1)
    
    line.set_data([0, x1, x2], [0, y1, y2])
    end_effector_path.set_data(Ex[:i], Ey[:i])
    return line,

#animates the plot
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(x), interval=1, blit=True)
#display the plot
plt.show()


np.savetxt('E:/Hackrush/level2.csv', np.column_stack((t[1:], Ex, Ey)), delimiter=',', header='Time,Ex,Ey', comments='')
#saves the animation as a video file
#anim.save('animation.mp4', fps=60, extra_args=['-vcodec', 'libx264'])