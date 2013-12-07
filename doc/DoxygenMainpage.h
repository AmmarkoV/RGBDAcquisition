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

@section libCompilation How to Compile It

Compilation of the library is pretty straightforward. 
You should enter the root directory of the project and run ./configure.sh 

The configure script should automatically prepare your working copy of RGBDAcquisition and prompt you to apt-get all the dependencies you might need , after those are installed it will prompt you to download 3dparty libraries ( like OpenNI , OpenNI2 , Freenect etc ) in order to further automate the process of configuring the lib in a new "vanilla" system.

After this is done just run make which should compile all of your libraries.

The make file should work in most cases but unfortunately due to the high number of external dependencies and configurations one can use the library with , things might become more complicated than this.

In case you want to see what is wrong you can download codeblocks and open the CodeBlocks.workspace in the root directory which will allow you to compile and "further investigate" your potential problems ( weird gcc version errors etc ) The makefile is produced using cbp2make ( http://sourceforge.net/projects/cbp2make/ ) so the codeblocks project is the source of everything.

CodeBlocks is an opensource IDE that is also used with wxSmith for making the GUI Editor Tool  

*/
