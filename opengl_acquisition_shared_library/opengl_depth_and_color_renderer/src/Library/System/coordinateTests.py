import numpy as np
import matplotlib.pyplot as plt
 
 

#---------------------------------------------------------
def create4x4FRotationX(degrees): 
    radians = degrees * ( np.pi / 180.0 )
    cosX = np.cos(radians)
    sinX = np.sin(radians)
    m = np.array([
                   [1.0,       0.0,   0.0, 0.0], 
                   [0.0,      cosX,  sinX, 0.0], 
                   [0.0, -1.0*sinX,  cosX, 0.0],
                   [0.0,       0.0,   0.0, 1.0] 
                 ])
    return m
#---------------------------------------------------------
def create4x4FRotationY(degrees):
     radians = degrees * ( np.pi / 180.0 )
     cosY = np.cos(radians)
     sinY = np.sin(radians)
     m = np.array([
                   [cosY,       0.0,   -1.0 * sinY, 0.0], 
                   [0.0,        1.0,           0.0, 0.0], 
                   [sinY,       0.0,          cosY, 0.0],
                   [0.0,        0.0,           0.0, 1.0] 
                 ])
     return m
#---------------------------------------------------------
def create4x4FRotationZ(degrees):
    radians = degrees * ( np.pi / 180.0 )
    cosZ = np.cos(radians)
    sinZ = np.sin(radians)
    m = np.array([
                   [cosZ,       sinZ,    0.0, 0.0], 
                   [-1.0*sinZ,  cosZ,    0.0, 0.0], 
                   [0.0,         0.0,    1.0, 0.0],
                   [0.0,         0.0,    0.0, 1.0] 
                 ])
    return m
#---------------------------------------------------------










 
fig = plt.figure()
ax = plt.axes(projection ='3d')


# defining all 3 axes
z = np.linspace(0, 1, 100)
x = z * np.sin(25 * z)
y = z * np.cos(25 * z)


z = [0, 0, 0]
x = [1, 1, 0]
y = [0, 0, 0]
ax.plot3D(x, y, z, 'red')

z = [1, 1, 0]
x = [0, 0, 0]
y = [0, 0, 0]
ax.plot3D(x, y, z, 'blue')

 
z = [0, 0, 0]
x = [0, 0, 0]
y = [1, 1, 0]
ax.plot3D(x, y, z, 'green')

# plotting
ax.set_title('3D line plot')
plt.show()

