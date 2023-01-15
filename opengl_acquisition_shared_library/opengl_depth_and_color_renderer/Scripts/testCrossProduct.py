import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#./BVHTester --from brokenHand.bvh --svg ./ --filterOccludedJoints


import sys
import os 
import numpy as np

def point_triangle_distance(point, triangle):
    # Define the triangle vertices
    A = np.array(triangle[0])
    B = np.array(triangle[1])
    C = np.array(triangle[2])

    # Define the triangle normal
    normal = np.cross(B - A, C - A)
    normal = normal / np.linalg.norm(normal)

    # Define the point vector
    P = np.array(point)

    # Calculate the distance between the point and the triangle
    distance = np.dot(normal, (P - A))

    # Calculate the side of the triangle
    side = np.dot(normal, np.cross(B - A, P - A))

    # Return the signed distance depending on the side
    if side > 0:
        return distance
    else:
        return -distance




#---------------------------------------------- 
point = [4.72,-7.19,-114.86] 
P0 = [-2.19,5.42,-126.98] 
P1 = [-8.60,-44.86,-141.50] 
P2 = [8.65,-43.18,-144.33] 
n = [0.18,-0.29,0.94] 
dist = 16.34 

#----------------------------------------------
triangle = [P0,P1,P2]
pyDist = point_triangle_distance(point,triangle) 
if (pyDist == dist ) : 
   print("Correct !!! ",pyDist)
else:
   print("Wrong !!! Py ",pyDist," vs C ",dist)


sys.exit(0)




# Define the vectors to plot
vectors = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Create a figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the vectors
for v in vectors:
    ax.quiver(0, 0, 0, v[0], v[1], v[2], color='r')

# Set the axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()
