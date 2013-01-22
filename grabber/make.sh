#!/bin/bash


cd ../acquisition
echo "Refreshing acquisition.so to reflect acquisition_setup.h"
make -f Acquisition.cbp.mak
cd ../grabber


OPENNI1_EXISTANCE=`cat ../acquisition/acquisition_setup.h | grep "#define USE_OPENNI1 1"` 
if [ -z "$OPENNI1_EXISTANCE" ] 
then
 echo "OpenNI1 not configured for use " 
 OPENNI1_LIBS=""
else
 echo "OpenNI1 activated " 
OPENNI1_LIBS="../3dparty/OpenNI-Bin-Dev-Linux-x64-v1.5.4.0/Lib/libOpenNI.so ../3dparty/OpenNI-Bin-Dev-Linux-x64-v1.5.4.0/Lib/libOpenNI.jni.so ../3dparty/OpenNI-Bin-Dev-Linux-x64-v1.5.4.0/Lib/libnimRecorder.so ../3dparty/OpenNI-Bin-Dev-Linux-x64-v1.5.4.0/Lib/libnimMockNodes.so ../3dparty/OpenNI-Bin-Dev-Linux-x64-v1.5.4.0/Lib/libnimCodecs.so ../openni1_acquisition_shared_library/libOpenNI1Acquisition.so"
fi



OPENNI2_EXISTANCE=`cat ../acquisition/acquisition_setup.h | grep "#define USE_OPENNI2 1"` 
if [ -z "$OPENNI2_EXISTANCE" ] 
then
 echo "OpenNI2 not configured for use " 
 OPENNI2_LIBS=""
else
 echo "OpenNI2 activated " 
 OPENNI2_LIBS="../3dparty/OpenNI-2.0.0/Redist/libOpenNI2.so ../openni2_acquisition_shared_library/libOpenNI2Acquisition.so"
fi


FREENECT_EXISTANCE=`cat ../acquisition/acquisition_setup.h | grep "#define USE_FREENECT 1"` 
if [ -z "$FREENECT_EXISTANCE" ] 
then
 echo "Freenect not configured for use " 
 FREENECT_LIBS=""
else
 echo "Freenect activated " 
 FREENECT_LIBS="../3dparty/libfreenect/build/lib/libfreenect_sync.so ../3dparty/libfreenect/build/lib/libfreenect.so ../libfreenect_acquisition_shared_library/libFreenectAcquisition.so"
fi


OPENGL_SANDBOX_EXISTANCE=`cat ../acquisition/acquisition_setup.h | grep "#define USE_OPENGL 1"` 
if [ -z "$OPENGL_SANDBOX_EXISTANCE" ] 
then
 echo "OpenGL sandbox not configured for use " 
 OPENGL_SANDBOX_LIBS=""
else
 echo "OpenGL sandbox activated " 
 OPENGL_SANDBOX_LIBS="-lGL -lX11 ../opengl_acquisition_shared_library/libOpenGLAcquisition.so"
fi


TEMPLATE_EXISTANCE=`cat ../acquisition/acquisition_setup.h | grep "#define USE_TEMPLATE 1"` 
if [ -z "$TEMPLATE_EXISTANCE" ] 
then
 echo "Template not configured for use " 
 TEMPLATE_LIBS=""
else
 echo "Template activated " 
 TEMPLATE_LIBS="../template_acquisition_shared_library/libTemplateAcquisition.so"
fi


ACQUISITION_LIBRARY="../acquisition/libAcquisition.so"


CFLAGS="-O3 -fexpensive-optimizations"
 
echo "LIBS TO LINK $OPENNI1_LIBS $OPENNI2_LIBS $FREENECT_LIBS $TEMPLATE_LIBS"

gcc -s main.c  $CFLAGS $ACQUISITION_LIBRARY $OPENNI1_LIBS $OPENNI2_LIBS $FREENECT_LIBS $OPENGL_SANDBOX_LIBS $TEMPLATE_LIBS -L. -o Grabber

exit 0
