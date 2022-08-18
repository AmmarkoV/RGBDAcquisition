import matplotlib.pyplot as plt
import numpy as np
import sys

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


jointName=" Title "
xLabel=" X "
yLabel=" Y "
zLabel=" Z "
viewAzimuth=45
viewElevation=45
transparency=0.8
scale = 0.15 #Smaller is smaller :P


if (len(sys.argv)>1):
       #print('Argument List:', str(sys.argv))
       for i in range(0, len(sys.argv)):
           if (sys.argv[i]=="--joint"):
             jointName=sys.argv[i+1]
           if (sys.argv[i]=="--x"):
             xLabel=sys.argv[i+1]
           if (sys.argv[i]=="--y"):
             yLabel=sys.argv[i+1]
           if (sys.argv[i]=="--z"):
             zLabel=sys.argv[i+1]
           if (sys.argv[i]=="--view"):
             viewAzimuth=int(sys.argv[i+1])
             viewElevation=int(sys.argv[i+2])


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

xs,ys,zs,vs = readData("study.dat")

maxValue = scale * max(vs)
svs = [(maxValue-x)/maxValue for x in vs]

ax.view_init(viewAzimuth,viewElevation) 

cm = plt.cm.get_cmap('RdYlBu')
sc = ax.scatter(xs, ys, zs, c=vs, s=svs, cmap=cm, alpha=transparency)
plt.colorbar(sc) 

ax.set_title(jointName) # Title of the plot
ax.set_xlabel(xLabel)
ax.set_ylabel(yLabel)
ax.set_zlabel(zLabel)

fig.savefig("out.png")
#plt.show()
