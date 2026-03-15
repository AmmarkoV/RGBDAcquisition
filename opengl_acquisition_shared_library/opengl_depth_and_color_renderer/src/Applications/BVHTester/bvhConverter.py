#!/usr/bin/python3

import ctypes
import os
import sys
from ctypes import *
from os.path import exists

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


#This mimics the calibration files like ;
# https://github.com/AmmarkoV/RGBDAcquisition/blob/master/tools/Calibration/calibration.c
def readCalibrationFromFile(filename):
    calib = dict()
    if filename is None:
        return calib

    fp = None
    try:
        fp = open(filename, "r")
    except IOError:
        return calib

    # Our state
    # ----------------------------
    i = 0
    category = 0
    line_length = 0
    lines_at_current_category = 0
    # ----------------------------


    for line in fp:
        #--------------------------------------
        line = line.rstrip("\r\n")
        line_length = len(line)
        #--------------------------------------
        if line_length > 0:
            if line[line_length - 1] == '\n':
                line = line[:-1]
            if line[line_length - 1] == '\r':
                line = line[:-1]
        #--------------------------------------
        if line_length > 1:
            if line[line_length - 2] == '\n':
                line = line[:-2]
            if line[line_length - 2] == '\r':
                line = line[:-2]
        #--------------------------------------
        if line[0] == '%':
            lines_at_current_category = 0
        #--------------------------------------
        # ---------------------------- ---------------------------- ----------------------------
        if line == "%I":
            category = 1
            calib["intrinsic"] = list()
        elif line == "%D":
            category = 2
        elif line == "%T":
            category = 3
            calib["extrinsicTranslation"] = list()
        elif line == "%R":
            category = 4
            calib["extrinsicRotationRodriguez"] = list()
        elif line == "%NF":
            category = 5
        elif line == "%UNIT":
            category = 6
        elif line == "%RT4*4":
            category = 7
            calib["extrinsic"] = list()
        elif line == "%Width":
            category = 8
        elif line == "%Height":
            category = 9
        else:
        # ---------------------------- ---------------------------- ----------------------------
            if category == 1:
                calib["intrinsicParametersSet"] = 1
                lines_at_current_category = min(lines_at_current_category, 9)
                calib["intrinsic"].append(float(line))
                lines_at_current_category += 1
                if (lines_at_current_category==9):
                                                   category = 0
            elif category == 2:
                if lines_at_current_category == 0:
                    calib["k1"] = float(line)
                elif lines_at_current_category == 1:
                    calib["k2"] = float(line)
                elif lines_at_current_category == 2:
                    calib["p1"] = float(line)
                elif lines_at_current_category == 3:
                    calib["p2"] = float(line)
                elif lines_at_current_category == 4:
                    calib["k3"] = float(line)
                lines_at_current_category += 1
                if (lines_at_current_category==4):
                                                   category = 0
            elif category == 3:
                calib["extrinsicParametersSet"] = 1
                lines_at_current_category = min(lines_at_current_category, 3)
                calib["extrinsicTranslation"].append(float(line))
                lines_at_current_category += 1
                if (lines_at_current_category==3):
                                                   category = 0
            elif category == 4:
                lines_at_current_category = min(lines_at_current_category, 3)
                calib["extrinsicRotationRodriguez"].append(float(line))
                lines_at_current_category += 1
                if (lines_at_current_category==3):
                                                   category = 0
            elif category == 5:
                calib["nearPlane"] = float(line)
                category = 0
            elif category == 6:
                calib["farPlane"] = float(line)
                category = 0
            elif category == 7:
                lines_at_current_category = min(lines_at_current_category, 16)
                calib["extrinsic"].append(float(line))
                lines_at_current_category += 1
                category = 0
            elif category == 8:
                    calib["width"] = int(line)
                    category = 0
            elif category == 9:
                    calib["height"] = int(line)
                    category = 0
        # ---------------------------- ---------------------------- ----------------------------

    fp.close()

    try:
        calib["fX"] = calib["intrinsic"][0]
        calib["fY"] = calib["intrinsic"][4]
        calib["cX"] = calib["intrinsic"][2]
        calib["cY"] = calib["intrinsic"][5]
    except:
        print("No intrinsic matrix declared in ", filename)
        print("Cannot populate fX, fY, cX, cY")


    print("New calibration loaded : ",calib)

    return calib



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
 libBVH.bvhConverter.restype  = c_int
 libBVH.bvhConverter.argtypes = c_int,POINTER(c_char_p)
 # context lifecycle
 libBVH.bvh_createContext.restype   = ctypes.c_void_p
 libBVH.bvh_createContext.argtypes  = []
 libBVH.bvh_destroyContext.restype  = None
 libBVH.bvh_destroyContext.argtypes = [ctypes.c_void_p]
 # load/unload — loadAtomic now returns the context pointer
 libBVH.bvhConverter_loadAtomic.restype  = ctypes.c_void_p
 libBVH.bvhConverter_loadAtomic.argtypes = [ctypes.c_char_p]
 libBVH.bvhConverter_unloadAtomic.restype  = ctypes.c_int
 libBVH.bvhConverter_unloadAtomic.argtypes = [ctypes.c_void_p]
 return libBVH
#--------------------------------------------------------


def splitDictionaryInLabelsAndFloats(arguments):
    #First prepare the labels of the joints we want to transmit
    #--------------------------------------------------
    labels = list(arguments.keys())
    labelsBytes = []
    for i in range(len(labels)):
        #Potential renaming..
        #---------------------------------------------
        #if ("endsite_" in labels[i]):
        #  if ("eye" in labels[i]):
        #   datasplit = labels[i].split("endsite_",1)
        #   newLabel="%s%s" % (datasplit[0],datasplit[1])
        #   print(labels[i]," renamed to -> ",newLabel)
        #---------------------------------------------
        labelsBytes.append(bytes(labels[i], 'utf-8'))
    labelsCStr = (ctypes.c_char_p * len(labelsBytes))()
    labelsCStr[:] = labelsBytes
    #--------------------------------------------------

    #Then prepare the array of floats we want to transmit
    #--------------------------------------------------
    values  = list(arguments.values())
    valuesF = list()
    for v in values:
       try:
         valuesF.append(float(v))
       except:
         print("Argument ",v,"cannot be casted to float..")
         valuesF.append(0.0)
    valuesArray    = (ctypes.c_float * len(valuesF))()
    valuesArray[:] = valuesF
    #--------------------------------------------------

    argc=len(labelsBytes)

    return labelsCStr,valuesArray,argc
#--------------------------------------------------------


class BVH():
  def __init__(
               self,
               bvhPath:str,
               libraryPath:str = "./libBVHConverter.so",
               cameraCalibrationFile = "",
               forceLibUpdate=False
              ):
        print("Initializing BVH file ",bvhPath," from ",libraryPath)
        self.libBVH               = loadLibrary(libraryPath,forceUpdate = forceLibUpdate)
        self.BVHContext           = None   # opaque context pointer returned by C
        self.numberOfJoints       = 0
        self.lastMAEErrorInPixels = 0.0
        self.traceStages          = False #If set to true each call will be emitted in stdout to speed-up debugging
        self.calib                = dict()
        #-----------------------------------
        if (cameraCalibrationFile!=""):
          if not exists(cameraCalibrationFile):
            print("Could not find renderer configuration file ",cameraCalibrationFile)
            raise FileNotFoundError
          # calibration is applied after load (which creates the context)
        #-----------------------------------
        if not exists(bvhPath):
            print("Could not find BVH file ",bvhPath)
            raise FileNotFoundError
        self.loadBVHFile(bvhPath)
        #-----------------------------------
        if (cameraCalibrationFile!=""):
          self.configureRendererFromFile(cameraCalibrationFile)
  #--------------------------------------------------------
  def stage(self,message):
        if (self.traceStages):
            print(bcolors.WARNING,message,bcolors.ENDC)
  #--------------------------------------------------------
  def loadBVHFile(self,bvhPath):
        self.stage("loadBVHFile")
        arg1 = bvhPath.encode('utf-8')
        # bvhConverter_loadAtomic allocates a new BVHContext and returns it as void*
        self.BVHContext = self.libBVH.bvhConverter_loadAtomic(arg1)
        if not self.BVHContext:
           print("Failed to load BVH file ",bvhPath)
           return 0
        self.libBVH.bvhConverter_getNumberOfJoints.argtypes = [ctypes.c_void_p]
        self.libBVH.bvhConverter_getNumberOfJoints.restype  = ctypes.c_int
        self.numberOfJoints = self.libBVH.bvhConverter_getNumberOfJoints(self.BVHContext)
        return self.numberOfJoints
  #--------------------------------------------------------
  def scale(self, scaleRatio:float):
        self.stage("scale")
        self.libBVH.bvhConverter_scale.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self.libBVH.bvhConverter_scale.restype  = ctypes.c_int
        return str(self.libBVH.bvhConverter_scale(self.BVHContext, scaleRatio))
  #--------------------------------------------------------
  def getJointName(self, jointID:int):
        self.stage("getJointName")
        self.libBVH.bvhConverter_getJointNameFromJointID.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.libBVH.bvhConverter_getJointNameFromJointID.restype  = ctypes.c_char_p
        return str(self.libBVH.bvhConverter_getJointNameFromJointID(self.BVHContext, jointID).decode('UTF-8'))
  #--------------------------------------------------------
  def isJointEndSite(self, jointID:int):
        self.stage("isJointEndSite")
        self.libBVH.bvhConverter_isJointEndSite.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.libBVH.bvhConverter_isJointEndSite.restype  = ctypes.c_int
        return self.libBVH.bvhConverter_isJointEndSite(self.BVHContext, jointID)
  #--------------------------------------------------------
  def getJointParent(self, jointID:int):
        self.stage("getJointParent")
        self.libBVH.bvhConverter_getJointParent.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.libBVH.bvhConverter_getJointParent.restype  = ctypes.c_int
        return self.libBVH.bvhConverter_getJointParent(self.BVHContext, jointID)
  #--------------------------------------------------------
  def getJointParentList(self):
        self.stage("getJointParentList")
        jointList = list()
        for jointID in range(0,self.numberOfJoints):
            jointList.append(int(self.getJointParent(jointID)))
        return jointList
  #--------------------------------------------------------
  def getMotionValueOfFrame(self, frameID:int, jointID:int):
        self.stage("getMotionValueOfFrame")
        self.libBVH.bvhConverter_getMotionValueOfFrame.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.libBVH.bvhConverter_getMotionValueOfFrame.restype  = ctypes.c_float
        return self.libBVH.bvhConverter_getMotionValueOfFrame(self.BVHContext, frameID, jointID)
  #--------------------------------------------------------
  def getAllMotionValuesOfFrame(self, frameID:int):
        allMIDs=list()
        for mID in range(0,self.getNumberOfMotionValuesPerFrame()):
          allMIDs.append(self.getMotionValueOfFrame(frameID,mID))
        return allMIDs
  #--------------------------------------------------------
  def saveBVHFileFromList(self, filename:str, allMotionData:list):
        self.stage("saveBVHFileFromList")
        arg1 = filename.encode('utf-8')
        #int bvhConverter_writeBVH(BVHHandle, char * filename,int writeHierarchy,int writeMotion)
        self.libBVH.bvhConverter_writeBVH.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        self.libBVH.bvhConverter_writeBVH.restype  = ctypes.c_int
        success = self.libBVH.bvhConverter_writeBVH(self.BVHContext, arg1, 1, 0) #Just write the Hierarchy part of the BVH file

        if (success):
         numberOfFrames = len(allMotionData)
         f = open(filename, 'a')
         f.write("MOTION\n");
         f.write("Frames: %u\n"%numberOfFrames);
         f.write("Frame Time: %0.8f\n"%(float(1/24)) );
         for fID in range(0,numberOfFrames):
          i=0
          for mID in allMotionData[fID]:
           if (i>0):
            f.write(' ')
           if (mID==0.0):
             f.write("0")
           else:
             f.write("%0.4f" % mID)
           i=i+1
          f.write('\n')
         f.close()

        #--------------------------------------------
        os.system("sed -i 's/rcollar/rCollar/g' out.bvh")
        os.system("sed -i 's/rshoulder/rShldr/g' out.bvh")
        os.system("sed -i 's/relbow/rForeArm/g' out.bvh")
        os.system("sed -i 's/rhand/rHand/g' out.bvh")
        #--------------------------------------------
        os.system("sed -i 's/lcollar/lCollar/g' out.bvh")
        os.system("sed -i 's/lshoulder/lShldr/g' out.bvh")
        os.system("sed -i 's/lelbow/lForeArm/g' out.bvh")
        os.system("sed -i 's/lhand/lHand/g' out.bvh")
        #--------------------------------------------
        os.system("sed -i 's/rhip/rThigh/g' out.bvh")
        os.system("sed -i 's/rknee/rShin/g' out.bvh")
        os.system("sed -i 's/rfoot/rFoot/g' out.bvh")
        #------------------------------------------------------
        os.system("sed -i 's/lhip/lThigh/g' out.bvh")
        os.system("sed -i 's/lknee/lShin/g' out.bvh")
        os.system("sed -i 's/lfoot/lFoot/g' out.bvh")

        return success
  #--------------------------------------------------------
  def setMotionValueOfFrame(self, frameID:int, jointID:int, value:float):
        self.stage("setMotionValueOfFrame")
        self.libBVH.bvhConverter_setMotionValueOfFrame.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float]
        self.libBVH.bvhConverter_setMotionValueOfFrame.restype  = ctypes.c_int
        return self.libBVH.bvhConverter_setMotionValueOfFrame(self.BVHContext, frameID, jointID, value)
  #--------------------------------------------------------
  def getNumberOfMotionValuesPerFrame(self):
        self.stage("getNumberOfMotionValuesPerFrame")
        self.libBVH.bvhConverter_getNumberOfMotionValuesPerFrame.argtypes = [ctypes.c_void_p]
        self.libBVH.bvhConverter_getNumberOfMotionValuesPerFrame.restype  = ctypes.c_int
        return self.libBVH.bvhConverter_getNumberOfMotionValuesPerFrame(self.BVHContext)
  #--------------------------------------------------------
  def getNumberOfJoints(self):
        self.stage("getNumberOfJoints")
        self.libBVH.bvhConverter_getNumberOfJoints.argtypes = [ctypes.c_void_p]
        self.libBVH.bvhConverter_getNumberOfJoints.restype  = ctypes.c_int
        return self.libBVH.bvhConverter_getNumberOfJoints(self.BVHContext)
  #--------------------------------------------------------
  def getJointID(self, jointName:str):
        self.stage("getJointID")
        arg1 = jointName.encode('utf-8')
        self.libBVH.bvhConverter_getJointNameJointID.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.libBVH.bvhConverter_getJointNameJointID.restype  = ctypes.c_int
        return self.libBVH.bvhConverter_getJointNameJointID(self.BVHContext, arg1)
  #--------------------------------------------------------
  def getJointList(self):
        self.stage("getJointList")
        jointList = list()
        for jointID in range(0,self.numberOfJoints):
            jointList.append(self.getJointName(jointID))
        return jointList
  #--------------------------------------------------------
  def getJointRotationsForFrame(self, jointID:int, frameID:int):
        self.stage("getJointRotationsForFrame")
        if (self.isJointEndSite(jointID)==1):
          xRot=0.0
          yRot=0.0
          zRot=0.0
        else:
          #--------------------------------------------------------
          self.libBVH.bvhConverter_getBVHJointRotationXForFrame.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
          self.libBVH.bvhConverter_getBVHJointRotationXForFrame.restype  = ctypes.c_float
          xRot = self.libBVH.bvhConverter_getBVHJointRotationXForFrame(self.BVHContext, frameID, jointID)
          #--------------------------------------------------------
          self.libBVH.bvhConverter_getBVHJointRotationYForFrame.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
          self.libBVH.bvhConverter_getBVHJointRotationYForFrame.restype  = ctypes.c_float
          yRot = self.libBVH.bvhConverter_getBVHJointRotationYForFrame(self.BVHContext, frameID, jointID)
          #--------------------------------------------------------
          self.libBVH.bvhConverter_getBVHJointRotationZForFrame.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
          self.libBVH.bvhConverter_getBVHJointRotationZForFrame.restype  = ctypes.c_float
          zRot = self.libBVH.bvhConverter_getBVHJointRotationZForFrame(self.BVHContext, frameID, jointID)
          #--------------------------------------------------------
        return xRot,yRot,zRot
  #--------------------------------------------------------
  def getJoint3D(self, jointID:int):
        self.stage("getJoint3D")
        self.libBVH.bvhConverter_get3DX.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.libBVH.bvhConverter_get3DX.restype  = ctypes.c_float
        x3D = self.libBVH.bvhConverter_get3DX(self.BVHContext, jointID)

        self.libBVH.bvhConverter_get3DY.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.libBVH.bvhConverter_get3DY.restype  = ctypes.c_float
        y3D = self.libBVH.bvhConverter_get3DY(self.BVHContext, jointID)

        self.libBVH.bvhConverter_get3DZ.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.libBVH.bvhConverter_get3DZ.restype  = ctypes.c_float
        z3D = self.libBVH.bvhConverter_get3DZ(self.BVHContext, jointID)

        return x3D,y3D,z3D
  #--------------------------------------------------------
  def getJoint2D(self, jointID:int):
        self.stage("getJoint2D")
        self.libBVH.bvhConverter_get2DX.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.libBVH.bvhConverter_get2DX.restype  = ctypes.c_float
        x2D = self.libBVH.bvhConverter_get2DX(self.BVHContext, jointID)

        self.libBVH.bvhConverter_get2DY.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.libBVH.bvhConverter_get2DY.restype  = ctypes.c_float
        y2D = self.libBVH.bvhConverter_get2DY(self.BVHContext, jointID)

        #Flip X
        if (x2D!=0.0) or (y2D!=0.0):
           x2D = 1.0 - x2D

        return x2D,y2D
  #--------------------------------------------------------
  def getJoint3DUsingJointName(self, jointName:str):
        return self.getJoint3D(self.getJointID(jointName))
  #--------------------------------------------------------
  def getJoint2DUsingJointName(self, jointName:str):
        return self.getJoint2D(self.getJointID(jointName))
  #--------------------------------------------------------
  def processFrame(self, frameID:int):
        self.stage("processFrame")
        self.libBVH.bvhConverter_processFrame.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.libBVH.bvhConverter_processFrame.restype  = ctypes.c_int
        return self.libBVH.bvhConverter_processFrame(self.BVHContext, frameID)
  #--------------------------------------------------------
  def modify(self,arguments:dict,frameID=0):
    self.stage("modify")
    #print("BVH modify called with : ",arguments)
    if (not arguments):
        print("BVH modify called without arguments")
        return 0
    #Arguments is a dict with a lot of key/value pairs we want to transmit to the C code
    labelsCStr,valuesArray,argc = splitDictionaryInLabelsAndFloats(arguments)
    self.libBVH.bvhConverter_modifyAtomic.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
    return self.libBVH.bvhConverter_modifyAtomic(self.BVHContext, labelsCStr, valuesArray, argc, frameID)
  #--------------------------------------------------------
  def configureRenderer(self,arguments:dict):
    #Arguments is a dict with a lot of key/value pairs we want to transmit to the C code
    labelsCStr,valuesArray,argc = splitDictionaryInLabelsAndFloats(arguments)
    self.libBVH.bvhConverter_rendererConfigurationAtomic.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    self.libBVH.bvhConverter_rendererConfigurationAtomic(self.BVHContext, labelsCStr, valuesArray, argc)
  #--------------------------------------------------------
  def configureRendererFromFile(self,cameraCalibrationFile:str):
    #from calibration import readCalibrationFromFile
    self.calib = readCalibrationFromFile(cameraCalibrationFile)
    if (self.calib):
               print("We found a calibration in file ",cameraCalibrationFile)
               print("calib : ",self.calib)
               self.configureRenderer(self.calib)
  #--------------------------------------------------------
  def get2DAnd3DAndBVHDictsForFrame(self,frameID=0):
    self.stage("get2DAnd3DAndBVHDictsForFrame ")
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
               #print("joint ID = ",jointID)
               #-------------------------------------------
               jointName = self.getJointName(jointID).lower()
               #-------------------------------------------
               #print("Getting 3D")
               x3D,y3D,z3D = self.getJoint3D(jointID)
               data3D["3DX_"+jointName]=float(x3D)
               data3D["3DY_"+jointName]=float(y3D)
               data3D["3DZ_"+jointName]=float(z3D)
               #-------------------------------------------
               #print("Getting 2D")
               x2D,y2D = self.getJoint2D(jointID)
               data2D["2DX_"+jointName]=float(x2D)
               data2D["2DY_"+jointName]=float(y2D)
               #-------------------------------------------
               #print("Getting Joint Rotations")
               if (self.isJointEndSite(jointID)==0): #Do not try to recover rotations for EndSites (they dont have rotations)
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
  def fineTuneToMatch(self,bodyPart:str,target:dict,frameID=0,iterations=20,epochs=30,lr=0.01,fSampling=30.0,fCutoff=5.0,langevinDynamics=0.0):
    self.stage("fineTuneToMatch ")
    bodyPartCStr = bytes(bodyPart, 'utf-8')

    #Arguments is a dict with a lot of key/value pairs we want to transmit to the C code
    labelsCStr,valuesArray,argc = splitDictionaryInLabelsAndFloats(target)
    self.libBVH.bvhConverter_IKFineTune.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
    self.libBVH.bvhConverter_IKFineTune.restype  = ctypes.c_float
    accuracy2D = self.libBVH.bvhConverter_IKFineTune(self.BVHContext, bodyPartCStr, labelsCStr, valuesArray, argc, frameID, iterations, epochs, lr, fSampling, fCutoff, langevinDynamics)
    #print("HCD results for ",iterations," iterations ~> %0.2f pixels!" % accuracy2D)
    self.lastMAEErrorInPixels = accuracy2D
    return self.get2DAnd3DAndBVHDictsForFrame(frameID=frameID)

   #return dict()
  #--------------------------------------------------------
  def smooth(self,frameID=0,fSampling=30.0,fCutoff=5.0):
    self.stage("smooth ")
    #This call assumes that is called after subsequent(?) calls to fineTuneToMatch that have transmitted the BVH state..!
    self.libBVH.bvhConverter_smooth.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_float]
    self.libBVH.bvhConverter_smooth.restype  = ctypes.c_int
    return self.libBVH.bvhConverter_smooth(self.BVHContext, frameID, fSampling, fCutoff) == 1

   #return dict()
  #--------------------------------------------------------
  def eraseHistory(self):
    self.stage("erase ")
    #This call assumes that is called after subsequent(?) calls to fineTuneToMatch that have transmitted the BVH state..!
    self.libBVH.bvhConverter_eraseHistory.argtypes = [ctypes.c_void_p, ctypes.c_int]
    self.libBVH.bvhConverter_eraseHistory.restype  = ctypes.c_int
    return self.libBVH.bvhConverter_eraseHistory(self.BVHContext, 0) == 1

   #return dict()
  #--------------------------------------------------------
  def __del__(self):
    if hasattr(self, 'libBVH') and self.libBVH is not None:
        if hasattr(self, 'BVHContext') and self.BVHContext is not None:
            self.libBVH.bvhConverter_unloadAtomic(self.BVHContext)
            self.BVHContext = None
  #--------------------------------------------------------



if __name__== "__main__":
   bvhFile = BVH(bvhPath="./headerWithHeadAndOneMotion.bvh",forceLibUpdate=True)

   print("File has ",bvhFile.numberOfJoints," joints")

   print(" Joint List : ",bvhFile.getJointList())
   print(" Joint Parent List : ",bvhFile.getJointParentList())

   modifications = dict()
   modifications["hip_Xposition"]=100.0
   modifications["hip_Yposition"]=100.0
   modifications["hip_Zposition"]=-400.0
   modifications["hip_Xrotation"]=1.0
   modifications["hip_Yrotation"]=2.0
   modifications["hip_Zrotation"]=4.0
   bvhFile.modify(modifications)
   jointName = "neck"
   print("Joint ID for ",jointName," is ",bvhFile.getJointID(jointName))

   frameID=0

   for i in range(0,10):
     modifications["hip_Xposition"]=100.0 + i * 10.0
     bvhFile.modify(modifications)
     bvhFile.processFrame(frameID)
     x3D,y3D,z3D = bvhFile.getJoint3DUsingJointName(jointName)
     print(" I=",i," Joint=",jointName," 3D values for frame ",frameID," are ",x3D,",",y3D,",",z3D," ")

   x2D,y2D = bvhFile.getJoint2DUsingJointName(jointName)
   print(" Joint ",jointName," 2D values for frame ",frameID," are ",x2D,",",y2D)

   target2D = dict()
   target2D["2dx_head"]=0.4722689390182495
   target2D["2dy_head"]=0.1971915066242218
   target2D["visible_head"]=0.9999899864196777
   target2D["2dx_lshoulder"]=0.43696290254592896
   target2D["2dy_lshoulder"]=0.2820419669151306
   target2D["visible_lshoulder"]=0.9999864101409912
   target2D["2dx_rshoulder"]=0.5070507228374481
   target2D["2dy_rshoulder"]=0.2563856244087219
   target2D["visible_rshoulder"]=0.9998794794082642
   target2D["2dx_neck"]=0.47200681269168854
   target2D["2dy_neck"]=0.26921379566192627
   target2D["visible_neck"]=0.9999329447746277
   target2D["2dx_hip"]=0.4936618059873581
   target2D["2dy_hip"]=0.49167898297309875
   target2D["visible_hip"]=0.9996901452541351

   print("fineTuneToMatch")
   result = bvhFile.fineTuneToMatch("body",target2D,frameID=0,iterations=10,epochs=30)
   #print("Result ",result)
