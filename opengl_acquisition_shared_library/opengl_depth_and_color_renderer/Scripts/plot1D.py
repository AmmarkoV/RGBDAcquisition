import matplotlib.pyplot as plt
import numpy as np
import sys

# Fixing random state for reproducibility
np.random.seed(19860811)

def readData(filename):
    linesRead = 0
    hasNonZeroElements=False
    columnA = list()
 
    fp = open(filename, 'r')
    lines = fp.readlines()

    for line in lines:
       if (linesRead>0): #<- skip label..!
         if (float(line)!=0.0):
          hasNonZeroElements=True
         columnA.append(float(line))
       linesRead = linesRead + 1
 
    return columnA,hasNonZeroElements

filename="study.dat"
output="out.png"
jointName=" Unknown Joint "
xLabel=" X "
yLabel=" Y "
zLabel=" Z "
viewAzimuth=45
viewElevation=45
transparency=0.8
scale = 0.12 #Smaller is smaller :P


if (len(sys.argv)>1):
       #print('Argument List:', str(sys.argv))
       for i in range(0, len(sys.argv)):
           if (sys.argv[i]=="--from"):
             filename=sys.argv[i+1]
           if (sys.argv[i]=="--to"):
             output=sys.argv[i+1]
             jointName=sys.argv[i+1]
           if (sys.argv[i]=="--joint"):
             jointName=sys.argv[i+1]


fig = plt.figure()

data,hasNonZeroElements = readData(filename)

if (hasNonZeroElements):
   plt.hist(data, bins=150)

   # Add labels and title
   plt.xlabel('Value')
   plt.ylabel('Frequency')
   plt.title('Histogram of %s '%jointName)

   # Save figure as PNG file
   plt.savefig(output)

   #ax.set_title(jointName) # Title of the plot 

   fig.savefig(output)
   #plt.show()
else: 
   print("File %s has only zero elements so not plotting anything.." % filename)



