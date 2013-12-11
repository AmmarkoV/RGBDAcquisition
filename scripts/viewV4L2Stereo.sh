#!/bin/bash 
clear
 

BINFOLDERPATH="."
BINPATH="run_viewer.sh -maxFrames 0 -module V4L2STEREO -i /dev/video1,/dev/video2 -o v4l2StereoTest -fps 30 $@"


if [ -e /usr/lib/x86_64-linux-gnu/libv4l/v4l2convert.so ]
then
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libv4l/v4l2convert.so $BINFOLDERPATH/$BINPATH
  exit $?
fi

if [ -e /usr/lib/libv4l/v4l2convert.so ]
then
  LD_PRELOAD=/usr/lib/libv4l/v4l2convert.so $BINFOLDERPATH/$BINPATH
  exit $?
fi

if [ -e /usr/lib32/libv4l/v4l2convert.so ]
then
  LD_PRELOAD=/usr/lib32/libv4l/v4l2convert.so $BINFOLDERPATH/$BINPATH
  exit $?
fi

if [ -e /usr/lib/i386-linux-gnu/libv4l/v4l2convert.so ]
then
  LD_PRELOAD=/usr/lib/i386-linux-gnu/libv4l/v4l2convert.so $BINFOLDERPATH/$BINPATH
  exit $?
fi
 
echo "Oh no.. they must have changed filenames again .. or you are running this script in an untested distro"
echo "That's why this script failed running $BINPATH in v4l2 compatibility mode"
echo "please try using *find -n v4l2convert.so* and update this compatibility script..!"

exit $?
   
