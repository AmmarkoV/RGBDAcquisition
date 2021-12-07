import numpy as np
import matplotlib.pyplot as plt
 
 
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
ax.plot3D(x, y, z, 'green')

 
z = [0, 0, 0]
x = [0, 0, 0]
y = [1, 1, 0]
ax.plot3D(x, y, z, 'blue')

# plotting
ax.set_title('3D line plot')
plt.show()

