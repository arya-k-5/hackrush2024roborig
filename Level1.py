import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
axis = plt.axes(xlim=(-5, 5), ylim=(-5, 5))
line, = axis.plot([], [], lw=2)

x = np.zeros((2, 2))
x[0, 0] = 0
x[0, 1] = 3
x[1, 0] = -1.75
x[1, 1] = 2

l1 = 1
l2 = 2

# Define joint angles for the two frames
theta1_values = [0, 1.5888404963642047]  # Replace with actual angles
theta2_values = [0, 1.0290593912966945]  # Replace with actual angles

def init():
    line.set_data([], [])
    return line,

def update(frame):
    theta1 = theta1_values[frame]
    theta2 = theta2_values[frame]
    
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    
    line.set_data([0, x1, x2], [0, y1, y2])
    return line,

ani = animation.FuncAnimation(fig, update, frames=2, init_func=init, blit=True)
plt.show()
