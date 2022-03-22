#!/usr/bin/python3

import ctypes
import os
import sys
from ctypes import *
from os.path import exists


def loadLibrary(filename):
#--------------------------------------------------------
 if not exists(filename):
     print("Could not find BVH Library (",filename,"), compiling a fresh one..!")
     print("Current directory was (",os.getcwd(),") ")
     directory=os.path.dirname(os.path.abspath(filename))
     os.system(directory+"/makeLibrary.sh")
 if not exists(filename):
     print("Could not make BVH Library, terminating")
     sys.exit(0)
 libBVH = CDLL(filename)
 #call C function to check connection
 libBVH.connect() 
 libBVH.bvhConverter.restype = c_int
 libBVH.bvhConverter.argtypes = c_int,POINTER(c_char_p)
 return libBVH
#--------------------------------------------------------




#pyarr = [0.1, 0.2, 0.3, 0.4]
#arr = (ctypes.c_float * len(pyarr))(*pyarr)










class BVH():
  def __init__(self, bvhPath:str, libraryPath:str = "./libBVHConverter.so" ):
        print("Initializing BVH from ",libraryPath)
        self.libBVH = loadLibrary(libraryPath)
        self.loadBVHFile(bvhPath)


  def loadBVHFile(self,bvhPath):
        # create byte objects from the strings
        arg1 = bvhPath.encode('utf-8')
 
        # send strings to c function
        self.libBVH.bvhConverter_loadAtomic.argtypes = [ctypes.c_char_p]
        self.libBVH.bvhConverter_loadAtomic(arg1)

  def modify(self,arguments:dict):
    #Arguments is a dict with a lot of key/value pairs we want to transmit to the C code
    
    #First prepare the labels of the joints we want to transmit
    #-------------------------------------------------- 
    labels = list(arguments.keys())
    labelsBytes = []
    for i in range(len(labels)):
        labelsBytes.append(bytes(labels[i], 'utf-8'))
    labelsCStr = (ctypes.c_char_p * len(labelsBytes))()
    labelsCStr[:] = labelsBytes 
    #-------------------------------------------------- 
    
    #Then prepare the array of floats we want to transmit
    #-------------------------------------------------- 
    values  = list(arguments.values())
    valuesF = list()
    for v in values:
        valuesF.append(float(v))
    valuesArray    = (ctypes.c_float * len(valuesF))()
    valuesArray[:] = valuesF
    #--------------------------------------------------  

    argc=len(labelsBytes)
    self.libBVH.bvhConverter_modifyAtomic.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    self.libBVH.bvhConverter_modifyAtomic(labelsCStr,valuesArray,argc)

if __name__== "__main__": 
   bvhFile = BVH(bvhPath="./headerWithHeadAndOneMotion.bvh") 
   modifications = dict()
   modifications["hip_Xposition"]=100.0
   modifications["hip_Yposition"]=200.0
   modifications["hip_Zposition"]=400.0
   bvhFile.modify(modifications)


