#!/bin/bash

THISDIR="acquisitionBroadcast"
THISOUT="acquisitionBroadcast"

if [ -d "AmmarServer/" ] 
then
 cd AmmarServer
 ./make
 cd .. 
fi


AMMAR_SERVER_LIBS=""
if [ -e "AmmarServer/src/AmmServerlib/libAmmServerlib.a" ] 
then
 AMMAR_SERVER_LIBS="AmmarServer/src/AmmServerlib/libAmmServerlib.a"
elif [ -e "AmmarServer/src/AmmServerNULLlib/libAmmServerlib.a" ] 
then
 AMMAR_SERVER_LIBS="AmmarServer/src/AmmServerNULLlib/libAmmServerlib.a"
fi




cd ../acquisition
echo "Refreshing acquisition.so to reflect acquisition_setup.h"
make -f Acquisition.cbp.mak
cd ../$THISDIR


OPENNI1_EXISTANCE=`cat ../acquisition/acquisition_setup.h | grep "#define USE_OPENNI1 1"` 
if [ -z "$OPENNI1_EXISTANCE" ] 
then
 echo "OpenNI1 not configured for use " 
 OPENNI1_LIBS=""
else
 echo "OpenNI1 activated " 

 cd ../openni1_acquisition_shared_library
 make -f OpenNI1Aquisition.cbp.mak
 cd ../$THISDIR

  OPENNI1_DIR="../3dparty/OpenNI/Platform/Linux/Bin/x64-Release"
OPENNI1_LIBS="$OPENNI1_DIR/libOpenNI.so $OPENNI1_DIR/libOpenNI.jni.so $OPENNI1_DIR/libnimRecorder.so $OPENNI1_DIR/libnimMockNodes.so $OPENNI1_DIR/libnimCodecs.so ../openni1_acquisition_shared_library/libOpenNI1Acquisition.so"
fi



OPENNI2_EXISTANCE=`cat ../acquisition/acquisition_setup.h | grep "#define USE_OPENNI2 1"` 
if [ -z "$OPENNI2_EXISTANCE" ] 
then
 echo "OpenNI2 not configured for use " 
 OPENNI2_LIBS=""
else
 echo "OpenNI2 activated " 

 cd ../openni2_acquisition_shared_library
 make -f OpenNI2Aquisition.cbp.mak
 cd ../$THISDIR

 OPENNI2_LIBS="../3dparty/OpenNI2/Bin/x64-Release/libOpenNI2.so ../openni2_acquisition_shared_library/libOpenNI2Acquisition.so"
fi


FREENECT_EXISTANCE=`cat ../acquisition/acquisition_setup.h | grep "#define USE_FREENECT 1"` 
if [ -z "$FREENECT_EXISTANCE" ] 
then
 echo "Freenect not configured for use " 
 FREENECT_LIBS=""
else
 echo "Freenect activated " 

 cd ../libfreenect_acquisition_shared_library
 make -f FreenectAcquisition.cbp.mak
 cd ../$THISDIR

 FREENECT_LIBS="../3dparty/libfreenect/build/lib/libfreenect_sync.so ../3dparty/libfreenect/build/lib/libfreenect.so ../libfreenect_acquisition_shared_library/libFreenectAcquisition.so"
fi


OPENGL_SANDBOX_EXISTANCE=`cat ../acquisition/acquisition_setup.h | grep "#define USE_OPENGL 1"` 
if [ -z "$OPENGL_SANDBOX_EXISTANCE" ] 
then
 echo "OpenGL sandbox not configured for use " 
 OPENGL_SANDBOX_LIBS=""
else
 echo "OpenGL sandbox activated " 

 cd ../opengl_acquisition_shared_library 
 make -f OpenGLAcquisition.cbp.mak
 cd ../$THISDIR

 OPENGL_SANDBOX_LIBS="-lGL -lX11 ../opengl_acquisition_shared_library/libOpenGLAcquisition.so"
fi


TEMPLATE_EXISTANCE=`cat ../acquisition/acquisition_setup.h | grep "#define USE_TEMPLATE 1"` 
if [ -z "$TEMPLATE_EXISTANCE" ] 
then
 echo "Template not configured for use " 
 TEMPLATE_LIBS=""
else
 echo "Template activated " 

 cd ../template_acquisition_shared_library
 make -f TemplateAcquisition.cbp.mak
 cd ../$THISDIR

 TEMPLATE_LIBS="../template_acquisition_shared_library/libTemplateAcquisition.so"
fi

 
ACQUISITION_LIBRARY="../acquisition/libAcquisition.so"


CFLAGS="-O3 -fexpensive-optimizations"
 
echo "LIBS TO LINK $OPENNI1_LIBS $OPENNI2_LIBS $FREENECT_LIBS $TEMPLATE_LIBS $AMMAR_SERVER_LIBS"

echo "gcc -s main.c  $CFLAGS $ACQUISITION_LIBRARY $OPENNI1_LIBS $OPENNI2_LIBS $FREENECT_LIBS $OPENGL_SANDBOX_LIBS $TEMPLATE_LIBS $AMMAR_SERVER_LIBS -lpthread -lrt -L. -o $THISOUT"

gcc -s main.c  $CFLAGS $ACQUISITION_LIBRARY $OPENNI1_LIBS $OPENNI2_LIBS $FREENECT_LIBS $OPENGL_SANDBOX_LIBS $TEMPLATE_LIBS $AMMAR_SERVER_LIBS -lpthread -lrt -L. -o $THISOUT



exit 0
