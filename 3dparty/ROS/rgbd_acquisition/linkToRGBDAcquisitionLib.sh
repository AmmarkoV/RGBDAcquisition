#!/bin/bash

#Switch to this directory
STARTDIR=`pwd` 
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
#-----------------------------------------------------------





# https://github.com/AmmarkoV/RGBDAcquisition <- this is the library that provides
# unified access  to OpenNI2 etc , to get it just execute 
# git clone git://github.com/AmmarkoV/RGBDAcquisition

# After you successfully compile it , please update the DATASETS path that follows and comment the next definition
#DATASETS_SEARCH="/home/ammar/Documents/Programming/RGBDAcquisition"

# Or let it be autoconfigured initially with uglier paths but i fix them..!
DATASETS_SEARCH="$DIR/../../../"

#Fix Dir to be prettier for bashrc etc..
cd $DATASETS_SEARCH
DATASETS=`pwd`
cd $DIR
#Ready to make links

cd src  
ln -s $DATASETS/acquisition/Acquisition.h
ln -s $DATASETS/tools/Calibration/calibration.h
cd ..



cd bin
#Copy things 
ln -s $DATASETS/acquisition/libAcquisition.so
ln -s $DATASETS/tools/Calibration/libCalibrationLibrary.a
ln -s $DATASETS/tools/OperatingSystem/libOperatingSystem.a
ln -s $DATASETS/tools/Codecs/libCodecs.a
ln -s $DATASETS/tools/Timers/libTimers.a
ln -s $DATASETS/tools/LocationServices/libLocationServices.a
ln -s $DATASETS/openni2_acquisition_shared_library/libOpenNI2Acquisition.so
ln -s $DATASETS/template_acquisition_shared_library/libTemplateAcquisition.so
ln -s $DATASETS/depthsense_acquisition_shared_library/libDepthSenseAcquisition.so 
ln -s $DATASETS/v4l2_acquisition_shared_library/libV4L2Acquisition.so
ln -s $DATASETS/v4l2stereo_acquisition_shared_library/libV4L2StereoAcquisition.so

ln -s $DATASETS/libfreenect_acquisition_shared_library/libFreenectAcquisition.so
ln -s  /usr/local/lib/libfreenect_sync.so.0.5
ln -s  /usr/local/lib/libfreenect.so.0.5

ln -s $DATASETS/librealsense_acquisition_shared_library/libRealsenseAcquisition.so
#ln -s  /usr/local/lib/libfreenect_sync.so.0.5
#ln -s  /usr/local/lib/libfreenect.so.0.5


ln -s $DATASETS/editor/Editor
ln -s $DATASETS/viewer/Viewer
 


if cat ~/.bashrc | grep -q "RGBDACQUISITION_PATH="
then
   echo "RGBDAcquisition PATH seems to be set up on bashrc!"  
else
   echo "Including RGBDAcquisition to .bashrc" 
   sh -c "echo \"export RGBDACQUISITION_PATH=$DATASETS\" >> ~/.bashrc"
   sh -c "echo \"export RGBDACQUISITION_REDIST=$DATASETS/redist\" >> ~/.bashrc"
   source ~/.bashrc
fi


#-----------------------------------------------------------
cd "$STARTDIR"


exit 0
