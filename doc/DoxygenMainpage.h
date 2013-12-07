/**
@mainpage  RGBDAcquisition
@author Ammar Qammaz a.k.a. AmmarkoV - http://ammar.gr

A uniform library wrapper for input from libfreenect,OpenNI,OpenNI2,OpenGL simulations and other types of video and depth input..

Website: https://github.com/AmmarkoV/RGBDAcquisition  

@section libIntro Introduction

 Any application that may want to interface with RGBDAcquition will probably want to link to libAcquisition.so
 and include this header. It provides the entry point for acquisition and internally loads/unloads all the existing
 sub-modules on runtime.
 Basic usage is the following

 We first want to initialize a module , lets say OPENNI2 with a maximum of 1 device and no settings. To do so we should call  
 
 acquisitionStartModule(ModuleIdentifier moduleID,unsigned int maxDevices,char * settings);  as
 
 with the appropriate arguments. After initializing the Module of our choice we are free to open a device for grabbing by using  

 acquisitionOpenDevice(ModuleIdentifier moduleID,DeviceIdentifier devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate);

 This is pretty much self documenting , so you just open a specific device of the module you have already initialized
 
 while (1) 
 {

  unsigned char * acquisitionGetColorFrame(ModuleIdentifier moduleID,DeviceIdentifier devID);

  unsigned short * acquisitionGetDepthFrame(ModuleIdentifier moduleID,DeviceIdentifier devID);

 }

 acquisitionCloseDevice(ModuleIdentifier moduleID,DeviceIdentifier devID);

 acquisitionStopModule(ModuleIdentifier moduleID);
  

@section libDesignOverview Design Overview

To do design overview
 See doc/Diagram.png

@section libWhyUseIt Why Use It ?

If you use linux and you want to take RGBD Input from kinect / openni / stereo  or another supported way quickly and efficiently without having to worry about all the details of each of the specific modules.
If you want a GUI to segment your dataset using depth bounding boxes planes , etc
If you want an OpenGL 3D renderer that can fit your datasets to do AR   
To get a simple input 

@section libCompilation How to Compile

Compilation of the library is pretty straightforward. 
You should enter the root directory of the project , run ./configure.sh and after this is done just run make which should in turn compile your library.

That should work in most cases but unfortunately due to the high number of external dependencies and configurations one can use the library with things might become more complicated than this  are not a


*/
