import ctypes
import os
import sys
from ctypes import *
from os.path import exists


def loadLibrary(filename):
 if not exists(filename):
     print("Could not find BVH Library, compiling a fresh one..!")
     os.system("./makeLibrary.sh")
 if not exists(filename):
     print("Could not make BVH Library, terminating")
     sys.exit(0)

 libBVH = CDLL(filename)
 #call C function to check connection
 libBVH.connect() 
 libBVH.bvhConverter.restype = c_int
 libBVH.bvhConverter.argtypes = c_int,POINTER(c_char_p)

 return libBVH

def bvhConvert(libBVH,arguments):
    argumentBytes = []
    for i in range(len(arguments)):
        argumentBytes.append(bytes(arguments[i], 'utf-8'))
    argv = (ctypes.c_char_p * len(argumentBytes))()
    argv[:] = argumentBytes 
    argc=len(argumentBytes)
    libBVH.bvhConverter(argc,argv)


libBVH = loadLibrary("./libBVHConverter.so")

startingFlags=list()


#Add any arguments given in the python script directly!
if (len(sys.argv)>1):
   print('Supplied argument List:', str(sys.argv))
   for i in range(0, len(sys.argv)):
     startingFlags.append(sys.argv[i])

   
startingFlags.append("--test")
startingFlags.append("--test 123")
startingFlags.append("--printparams")
 
bvhConvert(libBVH,startingFlags)
 

