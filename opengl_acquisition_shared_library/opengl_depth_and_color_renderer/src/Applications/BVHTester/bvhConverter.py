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
        for jointID in range(0,self.numberOfJoints):
            jointList.append(int(self.getJointParent(jointID)))
        return jointList
  #--------------------------------------------------------
  def getJointID(self, jointName:str):
        arg1 = jointName.encode('utf-8') 
        self.libBVH.bvhConverter_getJointNameJointID.argtypes = [ctypes.c_char_p]
        self.libBVH.bvhConverter_getJointNameJointID.restype  = ctypes.c_int
        jointID = self.libBVH.bvhConverter_getJointNameJointID(arg1)
        return jointID
  #--------------------------------------------------------
  def getJointList(self):
        jointList = list() 
        for jointID in range(0,self.numberOfJoints):
            jointList.append(self.getJointName(jointID))
        return jointList
  #--------------------------------------------------------
  def getJointRotationsForFrame(self, jointID:int, frameID:int):
        self.libBVH.bvhConverter_getBVHJointRotationXForFrame.argtypes = [ctypes.c_int, ctypes.c_int]
        self.libBVH.bvhConverter_getBVHJointRotationXForFrame.restype  = ctypes.c_float
        xRot = self.libBVH.bvhConverter_getBVHJointRotationXForFrame(jointID,frameID)

        self.libBVH.bvhConverter_getBVHJointRotationYForFrame.argtypes = [ctypes.c_int, ctypes.c_int]
        self.libBVH.bvhConverter_getBVHJointRotationYForFrame.restype  = ctypes.c_float
        yRot = self.libBVH.bvhConverter_getBVHJointRotationYForFrame(jointID,frameID)

        self.libBVH.bvhConverter_getBVHJointRotationZForFrame.argtypes = [ctypes.c_int, ctypes.c_int]
        self.libBVH.bvhConverter_getBVHJointRotationZForFrame.restype  = ctypes.c_float
        zRot = self.libBVH.bvhConverter_getBVHJointRotationZForFrame(jointID,frameID)

        return xRot,yRot,zRot 
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
        success = self.libBVH.bvhConverter_processFrame(frameID) 
        return success
  #--------------------------------------------------------
  def modify(self,arguments:dict,frameID=0):
    #Arguments is a dict with a lot of key/value pairs we want to transmit to the C code
    labelsCStr,valuesArray,argc = splitDictionaryInLabelsAndFloats(arguments)
    self.libBVH.bvhConverter_modifyAtomic.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
    success = self.libBVH.bvhConverter_modifyAtomic(labelsCStr,valuesArray,argc,frameID)
    return success
  #--------------------------------------------------------
  def configureRenderer(self,arguments:dict):
    #Arguments is a dict with a lot of key/value pairs we want to transmit to the C code
    labelsCStr,valuesArray,argc = splitDictionaryInLabelsAndFloats(arguments)
    self.libBVH.bvhConverter_rendererConfigurationAtomic.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    self.libBVH.bvhConverter_rendererConfigurationAtomic(labelsCStr,valuesArray,argc)
  #--------------------------------------------------------
  def get2DAnd3DAndBVHDictsForFrame(self,frameID=0):
    #Arguments is a dict with a lot of key/value pairs we want to transmit to the C code
    self.processFrame(frameID=frameID)

    #Our output     
    #---------------
    data2D  = dict()
    data3D  = dict()
    dataBVH = dict() 
    #---------------

    for jointID in range(0,self.numberOfJoints):
    #-------------------------------------------------------
               print("joint ID = ",jointID) 
               #-------------------------------------------
               jointName   = self.getJointName(jointID).lower()
               #-------------------------------------------
               x3D,y3D,z3D = self.getJoint3D(jointID)
               data3D["3DX_"+jointName]=float(x3D)
               data3D["3DY_"+jointName]=float(y3D)
               data3D["3DZ_"+jointName]=float(z3D)
               #-------------------------------------------
               x2D,y2D = self.getJoint2D(jointID)
               data2D["2DX_"+jointName]=float(x2D)
               data2D["2DY_"+jointName]=float(y2D)
               #-------------------------------------------
               xRot,yRot,zRot = self.getJointRotationsForFrame(jointID,frameID)
               if (jointID==0):
                  dataBVH[jointName+"_Xposition"]=float(x3D)
                  dataBVH[jointName+"_Yposition"]=float(y3D)
                  dataBVH[jointName+"_Zposition"]=float(z3D)
               dataBVH[jointName+"_Xrotation"]=float(xRot)
               dataBVH[jointName+"_Yrotation"]=float(yRot)
               dataBVH[jointName+"_Zrotation"]=float(zRot)
    #-------------------------------------------------------
    return data2D,data3D,dataBVH 
  #--------------------------------------------------------
  def fineTuneToMatch(self,bodyPart:str,target:dict,frameID=0):
    bodyPartCStr = bytes(bodyPart, 'utf-8')

    #Arguments is a dict with a lot of key/value pairs we want to transmit to the C code
    labelsCStr,valuesArray,argc = splitDictionaryInLabelsAndFloats(target)
    self.libBVH.bvhConverter_IKFineTune.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
    success = self.libBVH.bvhConverter_IKFineTune(bodyPartCStr,labelsCStr,valuesArray,argc,frameID)
    return success  
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

   target2D = dict()
   target2D["2DX_hip"]=100.0
   target2D["2DY_hip"]=200.0

   bvhFile.fineTuneToMatch("body",target2D,frameID=0)


