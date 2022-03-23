#!/usr/bin/python3

import ctypes
import os
import sys
from ctypes import *
from os.path import exists


def loadLibrary(filename,relativePath="",forceUpdate=False):
#--------------------------------------------------------
 if (relativePath!=""): 
     filename=relativePath+"/"+filename

 if (forceUpdate) or (not exists(filename)):
     print("Could not find BVH Library (",filename,"), compiling a fresh one..!")
     print("Current directory was (",os.getcwd(),") ")
     directory=os.path.dirname(os.path.abspath(filename))
     creationScript = directory+"/makeLibrary.sh"
     os.system(creationScript)
     #Magic JIT Just in time compilation, java has nothing on this :P 
 if not exists(filename):
     directory=os.path.dirname(os.path.abspath(filename))
     print("Could not make BVH Library, terminating")
     print("Directory we tried was : ",directory)
     sys.exit(0)
 libBVH = CDLL(filename)
 #call C function to check connection
 libBVH.connect() 
 libBVH.bvhConverter.restype = c_int
 libBVH.bvhConverter.argtypes = c_int,POINTER(c_char_p)
 return libBVH
#--------------------------------------------------------


def splitDictionaryInLabelsAndFloats(arguments):
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

    return labelsCStr,valuesArray,argc
#--------------------------------------------------------


class BVH():
  def __init__(self, bvhPath:str, libraryPath:str = "./libBVHConverter.so", forceLibUpdate=False ):
        print("Initializing BVH from ",libraryPath)
        self.libBVH         = loadLibrary(libraryPath,forceUpdate = forceLibUpdate)
        self.numberOfJoints = 0
        self.loadBVHFile(bvhPath)
  #--------------------------------------------------------
  def loadBVHFile(self,bvhPath):
        # create byte objects from the strings
        arg1 = bvhPath.encode('utf-8')
        # send strings to c function
        self.libBVH.bvhConverter_loadAtomic.argtypes = [ctypes.c_char_p]
        self.libBVH.bvhConverter_loadAtomic.restype  = ctypes.c_int
        self.numberOfJoints = self.libBVH.bvhConverter_loadAtomic(arg1)
        return self.numberOfJoints
  #--------------------------------------------------------
  def getJointName(self, jointID:int):
        self.libBVH.bvhConverter_getJointNameFromJointID.argtypes = [ctypes.c_int]
        self.libBVH.bvhConverter_getJointNameFromJointID.restype  = ctypes.c_char_p
        return str(self.libBVH.bvhConverter_getJointNameFromJointID(jointID).decode('UTF-8'));   
  #--------------------------------------------------------
  def getJointParent(self, jointID:int):
        self.libBVH.bvhConverter_getJointParent.argtypes = [ctypes.c_int]
        self.libBVH.bvhConverter_getJointParent.restype  = ctypes.c_int
        jointID = self.libBVH.bvhConverter_getJointParent(jointID)
        return jointID
  #--------------------------------------------------------
  def getJointParentList(self):
        jointList = list() 
        for jointID in range(0,bvhFile.numberOfJoints):
            jointList.append(int(bvhFile.getJointParent(jointID)))
        return jointList
  #--------------------------------------------------------
  def getJointID(self, jointName:str):
        arg1 = jointName.encode('utf-8') 
        self.libBVH.bvhConverter_getJointNameJointID.argtypes = [ctypes.c_char_p]
        jointID = self.libBVH.bvhConverter_getJointNameJointID(arg1)
        return jointID
  #--------------------------------------------------------
  def getJointList(self):
        jointList = list() 
        for jointID in range(0,bvhFile.numberOfJoints):
            jointList.append(bvhFile.getJointName(jointID))
        return jointList
  #--------------------------------------------------------
  def getJoint3D(self, jointID:int):
        self.libBVH.bvhConverter_get3DX.argtypes = [ctypes.c_int]
        self.libBVH.bvhConverter_get3DX.restype  = ctypes.c_float
        x3D = self.libBVH.bvhConverter_get3DX(jointID)

        self.libBVH.bvhConverter_get3DY.argtypes = [ctypes.c_int]
        self.libBVH.bvhConverter_get3DY.restype  = ctypes.c_float
        y3D = self.libBVH.bvhConverter_get3DY(jointID)

        self.libBVH.bvhConverter_get3DZ.argtypes = [ctypes.c_int]
        self.libBVH.bvhConverter_get3DZ.restype  = ctypes.c_float
        z3D = self.libBVH.bvhConverter_get3DZ(jointID)

        return x3D,y3D,z3D 
  #--------------------------------------------------------
  def getJoint2D(self, jointID:int):
        self.libBVH.bvhConverter_get2DX.argtypes = [ctypes.c_int]
        self.libBVH.bvhConverter_get2DX.restype  = ctypes.c_float
        x2D = self.libBVH.bvhConverter_get2DX(jointID)

        self.libBVH.bvhConverter_get2DY.argtypes = [ctypes.c_int]
        self.libBVH.bvhConverter_get2DY.restype  = ctypes.c_float
        y2D = self.libBVH.bvhConverter_get2DY(jointID)

        return x2D,y2D 
  #--------------------------------------------------------
  def getJoint3DUsingJointName(self, jointName:str):
        return self.getJoint3D(self.getJointID(jointName)) 
  #--------------------------------------------------------
  def getJoint2DUsingJointName(self, jointName:str):
        return self.getJoint2D(self.getJointID(jointName)) 
  #--------------------------------------------------------
  def processFrame(self, frameID:int):
        self.libBVH.bvhConverter_processFrame.argtypes = [ctypes.c_int]
        self.libBVH.bvhConverter_processFrame.restype = ctypes.c_int
        self.libBVH.bvhConverter_processFrame(frameID) 
  #--------------------------------------------------------
  def modify(self,arguments:dict,frameID=0):
    #Arguments is a dict with a lot of key/value pairs we want to transmit to the C code
    labelsCStr,valuesArray,argc = splitDictionaryInLabelsAndFloats(arguments)
    self.libBVH.bvhConverter_modifyAtomic.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
    self.libBVH.bvhConverter_modifyAtomic(labelsCStr,valuesArray,argc,frameID)
  #--------------------------------------------------------
  def configureRenderer(self,arguments:dict):
    #Arguments is a dict with a lot of key/value pairs we want to transmit to the C code
    labelsCStr,valuesArray,argc = splitDictionaryInLabelsAndFloats(arguments)
    self.libBVH.bvhConverter_rendererConfigurationAtomic.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    self.libBVH.bvhConverter_rendererConfigurationAtomic(labelsCStr,valuesArray,argc)
  #--------------------------------------------------------



if __name__== "__main__": 
   bvhFile = BVH(bvhPath="./headerWithHeadAndOneMotion.bvh",forceLibUpdate=True) 

   print("File has ",bvhFile.numberOfJoints," joints")

   print(" Joint List : ",bvhFile.getJointList())
   print(" Joint Parent List : ",bvhFile.getJointParentList())

   modifications = dict()
   modifications["hip_Xposition"]=100.0
   modifications["hip_Yposition"]=200.0
   modifications["hip_Zposition"]=400.0
   modifications["hip_Xrotation"]=1.0
   modifications["hip_Yrotation"]=2.0
   modifications["hip_Zrotation"]=4.0
   bvhFile.modify(modifications)
   jointName = "neck"
   print("Joint ID for ",jointName," is ",bvhFile.getJointID(jointName))

   frameID=0
   bvhFile.processFrame(frameID)
   x3D,y3D,z3D = bvhFile.getJoint3DUsingJointName(jointName)
   print(" Joint ",jointName," 3D values for frame ",frameID," are ",x3D,",",y3D,",",z3D," ")


   x2D,y2D = bvhFile.getJoint2DUsingJointName(jointName)
   print(" Joint ",jointName," 2D values for frame ",frameID," are ",x2D,",",y2D)

