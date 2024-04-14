import numpy as np
from numpy import sin, cos, tan, pi, sqrt, arccos, arctan2
from numpy.linalg import inv,det, pinv
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#constants

l1,l2 = 1,2
m1,m2 = 1,2
g = 9.81

def Jacobian(q1,q2):
    temp = np.array([
        [-l1*sin(q1) - l2*sin(q1+q2), - l2*sin(q1+q2)],
        [l1*cos(q1) +l2*cos(q1+q2), l2*cos(q1+q2)]
    ]) 

    return temp

def Jdot(q1,q2,q1_dot,q2_dot):
    J_dot = np.empty((2,2))
    J_dot[0,0] = -l1*np.sin(q1)*q1_dot - l2 * np.sin(q1+q2)*(q1_dot + q2_dot) 
    J_dot[0,1] = -l2*np.sin(q1+q2)*(q1_dot + q2_dot) 

    J_dot[1,0] = l1*np.cos(q1)*q1_dot + l2*np.cos(q1+q2)*(q1_dot + q2_dot) 
    J_dot[1,1] = l2*np.cos(q1+q2)*(q1_dot + q2_dot) 
    
    return J_dot

def MassMatrix(q1,q2):
    M = np.zeros((2,2)) 
    M[0,0] = m1*0.5*l1**2+ m2*(l1**2 + (0.5*l2)**2 + 2*l1*0.5*l2*cos(q2)) + (m1*l1**2)/12 + (m2*l2**2)/12
    M[0,1] = m2*((0.5*l2)**2 + l1*0.5*l2*cos(q2)) + (m2*l2**2)/12
    M[1,0] = M[0,1]
    M[1,1] = m2*(0.5*l2)**2+ (m2*l2**2)/12
    return M


def C(q1,q2,q1_dot,q2_dot):
    c = np.empty((2,2))
    h = -m2*l1*0.5*l2*sin(q2)
    c[0,0] = h*q2_dot
    c[0,1] = h*q2_dot+h*q1_dot
    c[1,0] = -h*q1_dot
    c[1,1] = 0
    return c
def G(q1,q2):
    grav = np.array([g*m1*l1/2*cos(q1) + g*m2*(l1*cos(q1) + 0.5*l2*cos(q1+q2)), g*m2*0.5*l2*cos(q1+q2)])
    return grav

def fk(q1,q2):
    x = l1*cos(q1) + l2*cos(q1+q2)
    y = l1*sin(q1) + l2*sin(q1+q2)
    return x,y

dt = 0.001
t = np.arange(0, 4, dt)

X = np.sin((np.pi/2)*t)
Y = np.cos((np.pi/2)*t)

Xd = pi*cos(pi*t/2)/2
Yd = -pi*sin(pi*t/2)/2

Xdd = -pi**2*sin(pi*t/2)/4
Ydd = -pi**2*cos(pi*t/2)/4

q = np.array([[0,0]]).T
q_dot = np.array([[0,0]]).T
q_dd = np.array([[0,0]]).T

Ex = []
Ey = []

kp = [[100,0],[0,100]]
kd = [[10,0],[0,10]]


Q1 = arccos((X**2 + Y**2 - l1**2 - l2**2)/(2*l1*l2))
Q2 = arctan2(Y,X)-arctan2(l2*sin(Q1),l1+l2*cos(Q1))

Q = np.array((Q1,Q2))

#inverse kinematics


for i in range(len(t)):
    J = Jacobian(q[0],q[1])
    J_dot = Jdot(q[0],q[1],q_dot[0],q_dot[1])
    M = MassMatrix(q[0],q[1])
    C = C(q[0],q[1],q_dot[0],q_dot[1])
    G = G(q[0],q[1])
    J_inv = pinv(J)
    Ex.append(X[i] - fk(q[0],q[1])[0])
    Ey.append(Y[i] - fk(q[0],q[1])[1])

    Qd = np.matmul(J_inv, np.array([[Xd[i]],[Yd[i]]]))
    Qdd = np.matmul(J_inv, np.array([[Xdd[i]], [Ydd[i] - np.matmul(J_dot, Qd)]]))

    Td = np.matmul(M,Qdd) + np.matmul(C,Qd) + G
    e = Q[:] - q[:]
    e_dot = Qd - q_dot[:]
    Tfb = np.matmul(kp,e) + np.matmul(kd,e_dot)
    Tc = Td + Tfb
    q_dd[:] = np.matmul(inv(M),Tc - np.matmul(C,q_dot[:,i]) - G)
    q_dot[:,i+1] = q_dot[:] + q_dd[:]*dt
    q[:,i+1] = q[:] + q_dot[:]*dt








