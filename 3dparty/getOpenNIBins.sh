#!/bin/bash

wget http://www.openni.org/openni-sdk/?download=http://www.openni.org/wp-content/uploads/2013/07/OpenNI-Linux-x64-2.2.tar.zip
unzip OpenNI-Linux-x64-2.2.tar.zip


wget http://www.openni.org/wp-content/uploads/2012/12/Sensor-Bin-Linux-x64-v5.1.2.1.tar.zip


wget http://www.openni.org/wp-content/uploads/2012/12/OpenNI-Bin-Dev-Linux-x64-v1.5.4.0.tar.zip



OPENNI2BINS="OpenNI-Linux-x64-2.2"

mkdir -p OpenNI2/Bin/x64-Release
cd OpenNI2/Bin/x64-Release
ln -s ../../../../$OPENNI2BINS/Redist/libOpenNI2.so
ln -s ../../../../$OPENNI2BINS/Redist/libOpenNI2.jni.so 
ln -s ../../../../$OPENNI2BINS/Redist/OpenNI2
ln -s ../../../../$OPENNI2BINS/Redist/OpenNI.ini

cd .. 
cd ..
cd ..



OPENNI1BINS="OpenNI-Bin-Dev-Linux-x64-v1.5.4.0"
mkdir -p  OpenNI/Platform/Linux/Bin/x64-Release
cd OpenNI/Platform/Linux/Bin/x64-Release

ln -s ../../../../$OPENNI1BINS/Lib/libnimCodecs.so
ln -s ../../../../$OPENNI1BINS/Lib/libnimRecorder.so
ln -s ../../../../$OPENNI1BINS/Lib/libOpenNI.jni.so
ln -s ../../../../$OPENNI1BINS/Lib/libOpenNI.so

  

cd .. 
cd ..
cd ..
cd ..
cd ..


