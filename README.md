![RGBDAcquisition](https://raw.githubusercontent.com/AmmarkoV/RGBDAcquisition/master/doc/imgBig.jpg)


# RGBDAcquisition Project
## A Linux RGBD image acquisition toolkit .. 

This is a collection of utilities that help me record and organize RGBD datasets for my computer vision needs.
The main feature is compatibility with a lot of different sensors and a streamlined approach to storing the datasets in order to segment them and use them for various experiments etc.  


## Building
------------------------------------------------------------------ 

To compile the library issue :

mkdir build 
cd build 
cmake .. 
make 


Since this library has plug-ins that are tied to V4L2,OpenNI,OpenNI2,Intel Realsense,Depthsense Kinetic,OpenGL,wxWidgets,OpenCV and other back-ends each has its own dependecies and when you first clone the repository and do this first build only the basic things will get built

If you want to download OpenNI , OpenNI2 , LibFreenect etc , please issue 
cd 3dparty && ./get_third_party_libs.sh

The script will ask you a series of questions and will try to download and setup the libraries you request.

if you wan to enable specific camera systems etc (and provided that their installations are ok) you can issue 

cd build 
cmake-gui ..

and there you will find a big list of ENABLE_X items  (ENABLE_OPENNI2,ENABLE_JPG,ENABLE_OPENGL etc ).
Switching them on and recompiling the library will generate the required plugins to use them.

There is also a GUI editor (ENABLE_EDITOR) based on wxWidgets that can be used as a graphical tool to acquire datasets, record them and then even segment them and do various simple processing tasks on them. 

after building it you can run it by issuing :
./run_editor.sh


The project is divided in libraries, applications, processors and tools


## Applications
------------------------------------------------------------------ 
- a Grabber , which can grab frames from the inputs and save them to disk 
- a Muxer   , which can combine 2 frame streams in to a new one and save it to disk
- a Broadcaster , which can transmit a stream through HTTP protocol
- a Viewer   , which can view a stream
- a Segmenter , which can discard or label parts of a stream
- an Editor , which is a graphical way to manage your datasets

## Tools
------------------------------------------------------------------ 
- Intrinsic / Extrinsic calibration using opencv
- Convertors from/to Euler/Quaternions
- Undistorting images ( intrinsic )
- Converting from 16bit PNG depths to PNM
- Calibrating a camera intrinsics/extrinsics using OpenCV 
- A ROS module to publish image streams using this library
- A programmable OpenGL scene simulator 
- A *Lot* of reusable code that does various simple computer vision tasks

## Libraries
------------------------------------------------------------------ 
- V4L2 acquisition
- V4L2 Stereo acquisition
- OpenNI1 acquisition
- OpenNI2 acquisition
- DepthSense Soft Kinetic acquisition
- Intel Realsense acquisition
- Desktop Recording (X11) Acquition
- libFreenect acquisition 
- OpenGL simulation acquistiion
- Template acquisition ( from images )

## Processors
------------------------------------------------------------------ 
- Darknet Neural Network processing
- Movidius/Intel Neural Compute Stick processing
- Simple obstacle detection 
 

When grabbing datasets you can select from the modules linked by using  them as a parameter..
For example
./run_grabber.sh -maxFrames 10 -module OPENNI2 -from 0 -to outdir 
this will grab 10 frames (-maxFrames 10) from the first(-from 0) OpenNI2 device (-module OPENNI2) plugged in ( provided it is compiled in ) and then write the output to grabber/frames/outdir

Possible choices for modules are :
V4L2
V4L2STEREO
FREENECT
OPENNI1
OPENNI2
OPENGL
TEMPLATE
NETWORK <-currently not working 
DEPTHSENSE 
REALSENSE 
DESKTOP 
SCRIPTED 


There are also processors that can process the acquired images 
For example assuming that you have a webcam connected, you have installed the darknet repository on the 3dparty directory and downloaded the appropriate weight files and you have also enabled the flags ENABLE_OPENCV, ENABLE_V4L2, ENABLE_PROCESSOR_DARKNET using cmake-gui you can process incoming images using darknet with the following command. 
  
./run_viewer.sh -module V4L2 -from /dev/video0  -noDepth -processor ../processors/DarknetProcessor/libDarknetProcessor.so  DarknetProcessor  $DIR/3dparty/darknet/yolo.weights $DIR/3dparty/darknet/cfg/yolo.cfg  $DIR/3dparty/darknet/cfg/coco.data --payload ./payload.sh $@

in order to dump 100 frames from your V4L2 Webcam to grabber/frames/v4l2test directory you can do

./run_grabber.sh -module V4L2 -from /dev/video0 -noDepth -maxFrames 100 -to v4l2test

----------------------------

In order to keep repository sizes small , and If you want to use acquisitionBroadcast you should run ./getBroadcastDependencies.sh to clone the full AmmarServer library which is not included since it has its own repository (https://github.com/AmmarkoV/AmmarServer/)
To test it try 
wget http://127.0.0.1:8080/rgb.ppm -O rgb.ppm
wget http://127.0.0.1:8080/depth.ppm -O depth.ppm
wget -qO- http://127.0.0.1:8080/control.html?seek=10


----------------------------


Without beeing 100% certain OpenNI installed via ROS may conflict with it being downloaded as a standalone package , there is a script that starts it the ROS way , so if thats your case just run scripts/ROS_StartCamera.sh and then you can start using OpenNI with no problems!
 
