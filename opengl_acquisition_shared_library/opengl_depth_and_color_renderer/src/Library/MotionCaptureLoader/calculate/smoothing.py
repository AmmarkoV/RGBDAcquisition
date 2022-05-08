#!/usr/bin/python3

import ctypes
import os
import sys
from ctypes import *
from os.path import exists



def loadLibrary(filename):
 if not exists(filename):
     print("Could not find Smoothing Library (",filename,"), compiling a fresh one..!")
     print("Current directory was (",os.getcwd(),") ")
     directory=os.path.dirname(os.path.abspath(filename))
     os.system(directory+"/makeLibrary.sh")
 if not exists(filename):
     print("Could not make Smoothing Library, terminating")
     sys.exit(0)
 libSmooth = CDLL(filename)
 #call C function to check connection
 libSmooth.connect() 
 libSmooth.bvhConverter.restype = c_int
 libSmooth.bvhConverter.argtypes = c_int,POINTER(c_char_p)
 return libBVH
#--------------------------------------------------------
def bvhConvert(libSmooth,arguments):
    argumentBytes = []
    for i in range(len(arguments)):
        argumentBytes.append(bytes(arguments[i], 'utf-8'))
    argv = (ctypes.c_char_p * len(argumentBytes))()
    argv[:] = argumentBytes 
    argc=len(argumentBytes)
    libBVH.bvhConverter(argc,argv)
#--------------------------------------------------------











if __name__== "__main__":
 #python main : 
 pythonFlags=list()
 #Add any arguments given in the python script directly!
 if (len(sys.argv)>1):
   print('Supplied argument List:', str(sys.argv))
   for i in range(0, len(sys.argv)):
     pythonFlags.append(sys.argv[i])
     if (sys.argv[i]=="--update"): 
         print('Deleting previous libSmoothing.so to force update!\n')
         os.system("rm libSmoothing.so")

 libSmooth = loadLibrary("./libSmoothing.so")
