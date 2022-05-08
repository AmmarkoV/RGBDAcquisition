#!/usr/bin/python3

import ctypes
import os
import sys
from ctypes import *
from os.path import exists


class Smooth():
  #--------------------------------------------------------
  def __init__(self, numberOfInputs:int, fSampling:float, fCutoff:float , libraryPath:str = "./libSmoothing.so", forceLibUpdate=False ):
     self.handle = 0
     if not exists(libraryPath):
          print("Could not find Smoothing Library (",libraryPath,"), compiling a fresh one..!")
          print("Current directory was (",os.getcwd(),") ")
          directory=os.path.dirname(os.path.abspath(libraryPath))
          os.system(directory+"/makeLibrary.sh")
     if not exists(libraryPath):
          print("Could not make Smoothing Library, terminating")
          sys.exit(0)
     self.numberOfInputs = numberOfInputs
     self.fSampling      = fSampling
     self.fCutoff        = fCutoff
     self.libSmooth = CDLL(libraryPath)
     #call C function to check connection
     self.libSmooth.connect() 
     self.libSmooth.butterWorth_allocateAtomic.restype   = ctypes.c_void_p
     self.libSmooth.butterWorth_allocateAtomic.argtypes  = [ ctypes.c_int, ctypes.c_float, ctypes.c_float ]
     self.handle = self.libSmooth.butterWorth_allocateAtomic(numberOfInputs,fSampling,fCutoff)
  #--------------------------------------------------------
  def filter(self,inputList:list):
        # create byte objects from the strings
        self.libSmooth.butterWorth_filterAtomic.restype  = ctypes.c_float
        self.libSmooth.butterWorth_filterAtomic.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]
          
        inputListLength = len(inputList)
        if (self.numberOfInputs!=inputListLength):
           print("Given %u inputs instead of expected %u, will not execute filter")
        else:
           for jID in range(0,inputListLength): 
             inputList[jID] = self.libSmooth.butterWorth_filterAtomic(self.handle,jID,inputList[jID])
        return inputList
  #--------------------------------------------------------
  def __del__(self):
     self.libSmooth.butterWorth_deallocateAtomic.restype  = ctypes.c_int
     self.libSmooth.butterWorth_deallocateAtomic.argtypes = [ ctypes.c_void_p ]
     self.libSmooth.butterWorth_deallocateAtomic(self.handle)
     self.handle = 0
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
  
 numberOfInputs = 10
 fSampling      = 120
 fCutoff        = 1.1  
 smooth = Smooth(numberOfInputs=numberOfInputs,fSampling=fSampling,fCutoff=fCutoff,libraryPath="./libSmoothing.so",forceLibUpdate = True)

 for frame in range(0,100):
   samples = list()
   for i in range(0,numberOfInputs): 
      samples.append(frame+1.0)
   print("Before -> ",samples)
   samples = smooth.filter(samples)
   print("After -> ",samples)






