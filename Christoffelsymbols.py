from sympy import symbols, Matrix, cos, sin , diff, zeros, Sum
import sympy as sp

# Define the symbols
q1, q2, q3,D,J1,J2,J3,I1 ,I2,I3,q,C,qd, q1d , q2d, q3d, cor, temp, Jacobian= symbols('q1 q2 q3 D J1 J2 J3 I1 I2 I3 q C qd q1d q2d q3d cor temp Jacobian')
t , x, y = symbols('t x y')

l1 = 2
l2 = 1
l3 = 1
m1 = 2
m2 =1
m3 = 1

x = sp.sin(0.5*sp.pi*t)
y = sp.cos(0.5*sp.pi*t)

x_dot = sp.diff(x,t)
y_dot = sp.diff(y,t)
x_ddot = sp.diff(x_dot,t)
y_ddot = sp.diff(y_dot,t)

q = [q1, q2, q3]
qd = [q1d, q2d, q3d]



#define jacobian 
def Jacobian(q1,q2,q3):
    temp = Matrix([
        [-l1*sin(q1) - l2*sin(q1+q2) - l3*sin(q1+q2+q3), - l2*sin(q1+q2)- l3*sin(q1+q2+q3),- l3*sin(q1+q2+q3)],
        [l1*cos(q1) +l2*cos(q1+q2)+l3*cos(q1+q2+q3) , l2*cos(q1+q2)+l3*cos(q1+q2+q3),l3*cos(q1+q2+q3)]
    ]) 




    return temp
print(Jacobian(q1,q2,q3))



# Define the matrices as functions of the given variables
def Jvc1(q1):
    return Matrix([
        [l1*0.5*sin(q1), 0,0],
        [l1*cos(q1)*0.5, 0,0],
        [0, 0,0]
    ])

def Jvc2(q1, q2):
    return Matrix([
        [-l1*sin(q1) - l2*sin(q1+q2)*0.5, - l2*sin(q1+q2)*0.5,0],
        [l1*cos(q1) +l2*cos(q1+q2)*0.5, l2*cos(q1+q2)*0.5,0],
        [0, 0,0]
    ])

def Jvc3(q1, q2, q3):

    return Matrix([
        [-l1*sin(q1) - l2*sin(q1+q2) - l3*0.5*sin(q1+q2+q3), - l2*sin(q1+q2)- l3*0.5*sin(q1+q2+q3),- l3*0.5*sin(q1+q2+q3)],
        [l1*cos(q1) +l2*cos(q1+q2)+l3*0.5*cos(q1+q2+q3) , l2*cos(q1+q2)+l3*0.5*cos(q1+q2+q3),l3*0.5*cos(q1+q2+q3)],
        [0, 0,0]
    ])

I1 = m1*l1**2/12
I2 = m2*l2**2/12
I3 = m3*l3**2/12

J1 = Jvc1(q1)
J2 = Jvc2(q1, q2)
J3 = Jvc3(q1, q2, q3)


D = m1 * J1.T * J1 + m2 * J2.T * J2 + m3 * J3.T * J3 + Matrix([[I1+I2+I3, I2+I3, I3], [I2+I3, I2+I3, I3], [I3, I3, I3]])


def Coriolis(D,q):
    temp = []
    for k in range(3):
        for j in range(3):
            for i in range(3):
                temp.append((0.5*(diff(D[k,j],q[i]) + diff(D[k,i],q[j]) - diff(D[i,j],q[k])))*qd[i]*qd[j])

    cor = [0,0,0]
    for i in range(9):
        cor[0] = cor[0]+temp[i]
    for i in range(9,18):
        cor[1] = cor[1]+temp[i]

    for i in range(18,27):
        cor[2] = cor[2]+temp[i]
    
    return cor


#define gravitational matrix

def Grav(q1,q2,q3):
    G = Matrix([
        [(m1*l1 + m2*l1 + m3*l1)*9.81*cos(q1) + (m2*l2 + m3*l2)*9.81*cos(q1+q2) + m3*l3*9.81*cos(q1+q2+q3)],
        [(m2*l2 + m3*l2)*9.81*cos(q1+q2) + m3*l3*9.81*cos(q1+q2+q3)],
        [m3*l3*9.81*cos(q1+q2+q3)]
    ])

    return G

