import numpy as np
import matplotlib.pyplot as plt
# from sympy import sin, cos ,symbols
import sympy as sy
from sympy import Matrix
import matplotlib.animation as animation


l1,l2 = 1, 2
m1,m2 = 1, 2

t,xd,yd,xddot,yddot,xddotdot,yddotdot,q1d,q2d,J,Jdot,qddot,qddotdot,Jd , Jinv, temp= sy.symbols("t xd yd xddot yddot xddotdot yddotdot q1d q2d J Jdot qddot qddotdot Jd Jinv temp")

xd = sy.sin((sy.pi/2)*t)
yd = sy.cos((sy.pi/2)*t)


xddot = sy.diff(xd,t)
yddot = sy.diff(yd,t)

xddotdot = sy.diff(xddot,t)
yddotdot = sy.diff(yddot,t)

q2d = sy.acos((xd**2 + yd**2 - l1**2 - l2**2)/(2*l1*l2))
q1d = sy.atan(yd/xd) - sy.atan((l2*sy.sin(q2d))/(l1+l2*sy.cos(q2d)))


J = Matrix([[-l1*sy.sin(q1d) - l2*sy.sin(q1d + q2d), -l2*sy.sin(q1d + q2d)],
     [ l1*sy.cos(q1d) + l2*sy.cos(q1d + q2d),  l2*sy.cos(q1d + q2d)]])

Jd = J.det()
Jinv = Matrix([[J[1,1],-J[0,1]],[-J[1,0],J[0,0]]])/Jd
#print(Jinv)

Jdot = sy.diff(J,t)

qddot = Jinv * Matrix([[xddot],[yddot]])

Xddotdot = Matrix([[xddotdot],[yddotdot]])
qddotdot = Jinv*(Xddotdot - Jdot*qddot)

print(qddotdot)





