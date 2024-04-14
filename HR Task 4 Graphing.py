import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv
from numpy.linalg import inv, det
from numpy import sin, cos, pi, tan
import matplotlib.animation as animation

import csv

file_path = r"C:\Users\vrula\Downloads\level4.csv"  # Using raw string literal

with open(file_path, 'r') as f:
    reader = csv.reader(f)
    columns_as_lists = [list(c) for c in zip(*reader)]

P = np.copy(columns_as_lists[1])
P = P[1:]
Q = np.copy(columns_as_lists[2])
Q = Q[1:]

l1 = 2
l2 = 1
l3 = 1
m1 = 2
m2 = 1
m3 = 1

#JAcobian matrix
def Jacobian(q1,q2,q3):
    J = np.empty((2,3))
    J[0,0] = -l1*np.sin(q1) - l2*np.sin(q1+q2) - l3*np.sin(q1+q2+q3)
    J[0,1] = -l2*np.sin(q1+q2) - l3*np.sin(q1+q2+q3)
    J[0,2] = -l3*np.sin(q1+q2+q3)
    J[1,0] = l1*np.cos(q1) + l2*np.cos(q1+q2) + l3*np.cos(q1+q2+q3)
    J[1,1] = l2*np.cos(q1+q2) + l3*np.cos(q1+q2+q3)
    J[1,2] = l3*np.cos(q1+q2+q3)

    return J

joint_angles = np.zeros((3, len(P)))

def func(theta1, theta2, theta3):
    J = Jacobian(theta1, theta2, theta3)
    X = l1*np.cos(theta1) + l2*np.cos(theta1+theta2) + l3*np.cos(theta1 + theta2 + theta3)
    Y = l1*np.sin(theta1) + l2*np.sin(theta1+theta2)+ l3*np.cos(theta1 + theta2 + theta3)
    return J, X, Y

Px = []
Py = []

for i in range(len(P)-1):
    theta1, theta2, theta3 = joint_angles[:, i]
    jacobian, x1, y1 = func(theta1, theta2, theta3)
    
    # Convert P[i] and Q[i] to float before subtraction
    E = np.array([float(P[i]), float(Q[i])]) - np.array([x1, y1])
    
    Px.append(x1)
    Py.append(y1) 
    joint_angles[:, i+1] = joint_angles[:, i] + pinv(jacobian).dot(E)

fig = plt.figure()
axis = plt.axes(xlim=(-5,5), ylim=(-5,5))
line, = axis.plot([], [], lw=2) #if there's an error later on, check this part
end_effector_path, = axis.plot(Px, Py, '--')

dt = 0.001
t = np.arange(0,4,dt)
theproperpathx = np.sin(pi*t/2)
theproperpathy = np.cos(pi*t/2)
plt.plot(theproperpathx, theproperpathy)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    theta1, theta2, theta3 = joint_angles[:, i]
    
    x1 = l1*np.cos(theta1)
    y1 = l1*np.sin(theta1)
    x2 = x1 + l2*np.cos(theta2+theta1)
    y2 = y1 + l2*np.sin(theta2+theta1)
    x3 = x2 + l3*np.cos(theta3 + theta2 + theta1)
    y3 = y2 + l3*np.sin(theta3 + theta2 + theta1)
    
    line.set_data([0, x1, x2, x3], [0, y1, y2, y3])
    
    # Update end effector path
    end_effector_path.set_data(Px[:i], Py[:i])
    
    return line,


#animates the plot
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(P), interval=1, blit=True)
#display the plot
plt.show()
#saves the animation as a video file
anim.save('animation.mp4', fps=60, extra_args=['-vcodec', 'libx264'])