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
point = [14.85,32.82,-120.67] 
P0 = [8.02,38.77,-158.73] 
P1 = [-0.82,-2.51,-190.33] 
P2 = [16.70,-1.84,-191.22] 
n = [0.06,-0.62,0.79] 

#----------------------------------------------
triangle = [P0,P1,P2]
print(point_triangle_distance(point,triangle))
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
