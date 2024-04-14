import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv
from numpy.linalg import inv, det
from numpy import sin, cos, pi, tan


# Constants

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

def Jdot(q1,q2,q3,q1_dot,q2_dot,q3_dot):
    J_dot = np.empty((2,3))
    J_dot[0,0] = -l1*np.sin(q1)*q1_dot - l2 * np.sin(q1+q2)*(q1_dot + q2_dot) - l3*np.sin(q1+q2+q3)*(q1_dot+ q2_dot + q3_dot)
    J_dot[0,1] = -l2*np.sin(q1+q2)*(q1_dot + q2_dot) - l3*np.sin(q1+q2+q3)*(q1_dot + q2_dot + q3_dot)
    J_dot[0,2] = -l3*np.sin(q1+q2+q3)*(q1_dot + q2_dot + q3_dot)
    J_dot[1,0] = l1*np.cos(q1)*q1_dot + l2*np.cos(q1+q2)*(q1_dot + q2_dot) + l3*np.cos(q1+q2+q3)*(q1_dot + q2_dot + q3_dot)
    J_dot[1,1] = l2*np.cos(q1+q2)*(q1_dot + q2_dot) + l3*np.cos(q1+q2+q3)*(q1_dot + q2_dot + q3_dot)
    J_dot[1,2] = l3*np.cos(q1+q2+q3)*(q1_dot + q2_dot + q3_dot)
    return J_dot





#Forward kinematics
def fk(q1,q2,q3):
    x = l1*np.cos(q1) + l2*np.cos(q1+q2) + l3*np.cos(q1+q2+q3)
    y = l1*np.sin(q1) + l2*np.sin(q1+q2) + l3*np.sin(q1+q2+q3)
    return x,y

#Mass matrix
def M(q1,q2,q3):
    Mat  = np.array([[0.25*l1**2*m1*sin(q1)**2 + 0.25*l1**2*m1*cos(q1)**2 + l1**2*m1/12 + l2**2*m2/12 + l3**2*m3/12 + m2*(-l1*sin(q1) - 0.5*l2*sin(q1 + q2))**2 + m2*(l1*cos(q1) + 0.5*l2*cos(q1 + q2))**2 + m3*(-l1*sin(q1) - l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))**2 + m3*(l1*cos(q1) + l2*cos(q1 + q2) + 0.5*l3*cos(q1 + q2 + q3))**2, l2**2*m2/12 - 0.5*l2*m2*(-l1*sin(q1) - 0.5*l2*sin(q1 + q2))*sin(q1 + q2) + 0.5*l2*m2*(l1*cos(q1) + 0.5*l2*cos(q1 + q2))*cos(q1 + q2) + l3**2*m3/12 + m3*(-l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))*(-l1*sin(q1) - l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3)) + m3*(l2*cos(q1 + q2) + 0.5*l3*cos(q1 + q2 + q3))*(l1*cos(q1) + l2*cos(q1 + q2) + 0.5*l3*cos(q1 + q2 + q3)), l3**2*m3/12 - 0.5*l3*m3*(-l1*sin(q1) - l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))*sin(q1 + q2 + q3) + 0.5*l3*m3*(l1*cos(q1) + l2*cos(q1 + q2) + 0.5*l3*cos(q1 + q2 + q3))*cos(q1 + 
        q2 + q3)], [l2**2*m2/12 - 0.5*l2*m2*(-l1*sin(q1) - 0.5*l2*sin(q1 + q2))*sin(q1 + q2) + 0.5*l2*m2*(l1*cos(q1) + 0.5*l2*cos(q1 + q2))*cos(q1 + q2) + l3**2*m3/12 + m3*(-l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))*(-l1*sin(q1) - l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3)) + m3*(l2*cos(q1 + q2) + 0.5*l3*cos(q1 + q2 + q3))*(l1*cos(q1) + l2*cos(q1 + q2) + 0.5*l3*cos(q1 + q2 + q3)), 0.25*l2**2*m2*sin(q1 + q2)**2 + 0.25*l2**2*m2*cos(q1 + q2)**2 + l2**2*m2/12 + l3**2*m3/12 + m3*(-l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))**2 
        + m3*(l2*cos(q1 + q2) + 0.5*l3*cos(q1 + q2 + q3))**2, l3**2*m3/12 - 0.5*l3*m3*(-l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))*sin(q1 + q2 + q3) + 0.5*l3*m3*(l2*cos(q1 + q2) + 0.5*l3*cos(q1 + q2 + q3))*cos(q1 + q2 + q3)], [l3**2*m3/12 - 0.5*l3*m3*(-l1*sin(q1) - l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))*sin(q1 + q2 + q3) + 0.5*l3*m3*(l1*cos(q1) + l2*cos(q1 + q2) + 0.5*l3*cos(q1 + q2 + q3))*cos(q1 + q2 + q3), l3**2*m3/12 - 0.5*l3*m3*(-l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))*sin(q1 + q2 + q3) + 0.5*l3*m3*(l2*cos(q1 + q2) + 0.5*l3*cos(q1 + q2 + q3))*cos(q1 + q2 + q3), 0.25*l3**2*m3*sin(q1 + q2 + q3)**2 + 0.25*l3**2*m3*cos(q1 + q2 + q3)**2 + l3**2*m3/12]])
    return Mat

#Coriolis + gravity matrix
def H(q1,q2,q3,q1d,q2d,q3d):
    C = np.empty((3,1))
    C[0] = q1d**2*(0.5*m2*(-2*l1*sin(q1) - 1.0*l2*sin(q1 + q2))*(l1*cos(q1) + 0.5*l2*cos(q1 + q2)) + 0.5*m2*(-l1*sin(q1) - 0.5*l2*sin(q1 + q2))*(-2*l1*cos(q1) - 1.0*l2*cos(q1 + q2)) + 0.5*m3*(-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 1.0*l3*sin(q1 + q2 + q3))*(l1*cos(q1) + l2*cos(q1 + q2) + 0.5*l3*cos(q1 + q2 + q3)) + 0.5*m3*(-l1*sin(q1) - l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))*(-2*l1*cos(q1) - 2*l2*cos(q1 + q2) - 1.0*l3*cos(q1 + q2 + q3))) + 2*q1d*q2d*(-0.5*l2*m2*(-l1*sin(q1) - 0.5*l2*sin(q1 + q2))*cos(q1 + q2) - 0.5*l2*m2*(l1*cos(q1) + 0.5*l2*cos(q1 + q2))*sin(q1 + q2) + 0.5*m3*(-2*l2*sin(q1 + q2) - 1.0*l3*sin(q1 + q2 + q3))*(l1*cos(q1) + l2*cos(q1 + q2) + 0.5*l3*cos(q1 + q2 + q3)) + 0.5*m3*(-2*l2*cos(q1 + q2) - 1.0*l3*cos(q1 + q2 + q3))*(-l1*sin(q1) - l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))) + 2*q1d*q3d*(-0.5*l3*m3*(-l1*sin(q1) - l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 0.5*l3*m3*(l1*cos(q1) + l2*cos(q1 + q2) + 0.5*l3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + q2d**2*(-0.5*l2*m2*(-l1*sin(q1) - 0.5*l2*sin(q1 + q2))*cos(q1 + q2) - 0.5*l2*m2*(l1*cos(q1) + 0.5*l2*cos(q1 + q2))*sin(q1 + q2) - 0.5*m3*(-2*l2*sin(q1 + q2) - 1.0*l3*sin(q1 + q2 + q3))*(l2*cos(q1 + q2) + 0.5*l3*cos(q1 + q2 + q3)) - 0.5*m3*(-l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))*(-2*l2*cos(q1 + q2) - 1.0*l3*cos(q1 + q2 + q3)) + 1.0*m3*(-l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))*(-l2*cos(q1 + q2) - 0.5*l3*cos(q1 + q2 + q3)) + 1.0*m3*(-l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))*(l2*cos(q1 + q2) + 0.5*l3*cos(q1 + q2 + q3)) + 1.0*m3*(-l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))*(l1*cos(q1) + l2*cos(q1 + q2) + 0.5*l3*cos(q1 + q2 + q3)) + 1.0*m3*(-l2*cos(q1 + q2) - 0.5*l3*cos(q1 + q2 + q3))*(-l1*sin(q1) - l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))) + 2*q2d*q3d*(-0.5*l3*m3*(-l1*sin(q1) - l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 0.5*l3*m3*(l1*cos(q1) + l2*cos(q1 + q2) + 0.5*l3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + q3d**2*(-0.5*l3*m3*(-l1*sin(q1) - l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 0.5*l3*m3*(l1*cos(q1) + l2*cos(q1 + q2) 
    + 0.5*l3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3))
    C[1] = q1d**2*(0.5*l2*m2*(-l1*sin(q1) - 0.5*l2*sin(q1 + q2))*cos(q1 + q2) - 0.5*l2*m2*(-l1*cos(q1) - 0.5*l2*cos(q1 + q2))*sin(q1 + q2) - 0.5*m3*(-2*l2*sin(q1 + q2) - 1.0*l3*sin(q1 + q2 + q3))*(l1*cos(q1) + l2*cos(q1 + q2) + 0.5*l3*cos(q1 + q2 + q3)) + 1.0*m3*(-l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))*(-l1*cos(q1) - l2*cos(q1 + q2) - 0.5*l3*cos(q1 + q2 + q3)) + 1.0*m3*(-l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))*(l1*cos(q1) + l2*cos(q1 + q2) + 0.5*l3*cos(q1 + q2 + q3)) - 0.5*m3*(-2*l2*cos(q1 + q2) - 1.0*l3*cos(q1 + q2 + q3))*(-l1*sin(q1) - l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3)) + 1.0*m3*(-l2*cos(q1 + q2) - 0.5*l3*cos(q1 + q2 + q3))*(-l1*sin(q1) - l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3)) + 1.0*m3*(l2*cos(q1 + q2) + 0.5*l3*cos(q1 + q2 + q3))*(-l1*sin(q1) - l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))) + 2*q1d*q2d*(0.5*m3*(-2*l2*sin(q1 + q2) - 1.0*l3*sin(q1 + q2 + q3))*(l2*cos(q1 + q2) + 0.5*l3*cos(q1 + q2 + q3)) + 0.5*m3*(-l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))*(-2*l2*cos(q1 + q2) - 1.0*l3*cos(q1 + q2 + q3))) + 2*q1d*q3d*(-0.5*l3*m3*(-l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 0.5*l3*m3*(l2*cos(q1 + q2) + 0.5*l3*cos(q1 + q2 + q3))*sin(q1 + q2 
    + q3)) + q2d**2*(0.5*m3*(-2*l2*sin(q1 + q2) - 1.0*l3*sin(q1 + q2 + q3))*(l2*cos(q1 + q2) + 0.5*l3*cos(q1 + q2 + q3)) + 0.5*m3*(-l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))*(-2*l2*cos(q1 + q2) - 1.0*l3*cos(q1 + q2 + q3))) + 2*q2d*q3d*(-0.5*l3*m3*(-l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 0.5*l3*m3*(l2*cos(q1 + q2) + 0.5*l3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + q3d**2*(-0.5*l3*m3*(-l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 0.5*l3*m3*(l2*cos(q1 + q2) + 0.5*l3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3))
    C[2] = q1d**2*(0.5*l3*m3*(-l1*sin(q1) - l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 0.5*l3*m3*(-l1*cos(q1) - l2*cos(q1 + q2) - 0.5*l3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 2*q1d*q2d*(0.5*l3*m3*(-l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 0.5*l3*m3*(-l2*cos(q1 + q2) - 0.5*l3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + q2d**2*(0.5*l3*m3*(-l2*sin(q1 + q2) - 0.5*l3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 0.5*l3*m3*(-l2*cos(q1 + q2) - 0.5*l3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3))

    G = np.zeros((3,1))
    G[0] = (m1*l1*0.5 + m2*l1 + m3*l1)*9.81*np.cos(q1) + (m2*l2*0.5 + m3*l2)*9.81*np.cos(q1+q2) + m3*l3*0.5*9.81*np.cos(q1+q2+q3)
    G[1] = (m2*l2*0.5 + m3*l2)*9.81*np.cos(q1+q2) + m3*l3*0.5*9.81*np.cos(q1+q2+q3)
    G[2] = m3*l3*0.5*9.81*np.cos(q1+q2+q3)

    H = C + G
    return H


#trajectory 
t = np.arange(0,4,0.001)
# desired trajectory x and y
X = np.sin(0.5*np.pi*t)
Y = np.cos(0.5*np.pi*t)
#desired x and y dot
Xdot = pi*cos(pi*t/2)/2
Ydot = -pi*sin(pi*t/2)/2
#desired xdot dot and dot dot
Xdd = -pi**2*sin(pi*t/2)/4
Ydd = -pi**2*cos(pi*t/2)/4



#initial conditions
qc = np.empty((3, len(t)))
qc[:,0] = [0.01,0.01,0.01]


Tc = np.empty((3, len(t)))
qdotc = np.empty((3, len(t)))
qdotc[:,0] = [0,0,0]

qddot = np.empty((3, len(t)))

# control gains 
kp = [[100,0,0],[0,100,0],[0,0,100]]
kd = [[10,0,0],[0,10,0],[0,0,10]]

Ex = []
Ey = []

for i in range(len(t)):
    J = Jacobian(qc[0,i],qc[1,i],qc[2,i])
    Jd = Jdot(qc[0,i],qc[1,i],qc[2,i],qdotc[0,i],qdotc[1,i],qdotc[2,i])
    xc,yc = fk(qc[0,i],qc[1,i],qc[2,i])
    X_ = np.array((2,1))
    X_[0] = X[i]
    X_[1] = Y[i]
    Ex.append(X_[0]-xc)
    Ey.append(X_[1]-yc)
    print(Ey[i])
    X_dot = np.array([[Xdot[i]],[Ydot[i]]])
    X_dd = np.array([[Xdd[i]],[Ydd[i]]])
    # print(det(np.matmul(J.T,J)))
    # print(pinv(J))
    result = np.matmul((np.identity(3) - np.matmul(pinv(J),J)),qdotc[:,i])
    result_reshaped = result.reshape((3, 1))
    Q_dot = np.matmul(pinv(J),X_dot) + result_reshaped
    qctemp = qc[:,i].reshape((3,1))
    qdotctemp = qdotc[:,i].reshape((3,1))
    Q = 0.001 * Q_dot  + qctemp
    Qddot = np.matmul(pinv(J),(X_dd - np.matmul(Jd,Q_dot)))
    Tff = np.matmul(M(qc[0,i],qc[1,i],qc[2,i]),Qddot) + H(qc[0,i],qc[1,i],qc[2,i],qdotc[0,i],qdotc[1,i],qdotc[2,i])
    e = Q - qctemp

    e_dot = Q_dot - qdotctemp

    Tfb = np.matmul(kp,e) + np.matmul(kd,e_dot)

    Tc[0,i] = (Tff + Tfb) [0,0]
    Tc[1,i] = (Tff + Tfb) [1,0]
    Tc[2,i] = (Tff + Tfb) [2,0]
    
    temp = Tc[:,i] - H(qc[0,i],qc[1,i],qc[2,i],qdotc[0,i],qdotc[1,i],qdotc[2,i])
    
    qddot[0,i+1] = (np.matmul(inv(M(qc[0,i],qc[1,i],qc[2,i])),temp))[0,0]
    qddot[1,i+1] = (np.matmul(inv(M(qc[0,i],qc[1,i],qc[2,i])),temp))[1,0]
    qddot[2,i+1] = (np.matmul(inv(M(qc[0,i],qc[1,i],qc[2,i])),temp))[2,0]
    qdotc[:,i+1] = qdotc[:,i] + qddot[:,i+1]*0.001
    qc[:,i+1] = qc[:,i] + qdotc[:,i+1]*0.001


