import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

def readData(filename):

    columnA = list()  
    columnB = list()  
    columnC = list()  
    columnD = list()  
 
    fp = open(filename, 'r')
    lines = fp.readlines()

    for line in lines:
       d = line.split(' ')
       columnA.append(float(d[0]))
       columnB.append(float(d[1]))
       columnC.append(float(d[2]))
       columnD.append(float(d[3]))
 
    return columnA, columnB, columnC, columnD



def randrange(n, vmin, vmax):
    """
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    """
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

xs,ys,zs,vs = readData("study.dat")
ax.scatter(xs, ys, zs, c=vs, s=0.5, alpha=0.5)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
