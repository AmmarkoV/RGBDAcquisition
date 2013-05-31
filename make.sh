#!/bin/bash

if [ -d openni1_acquisition_shared_library ]
then  
cd openni1_acquisition_shared_library
ln -s OpenNI1Acquisition.so libOpenNI1Acquisition.so
cd ..
fi

if [ -d openni2_acquisition_shared_library ]
then  
cd openni2_acquisition_shared_library
ln -s OpenNI2Acquisition.so libOpenNI2Acquisition.so
cd ..
fi

if [ -d grabber ]
then  
cd grabber
./make.sh
cd ..
fi

if [ -d acquisitionBroadcast ]
then  
cd acquisitionBroadcast
./make.sh
cd ..
fi

exit 0
