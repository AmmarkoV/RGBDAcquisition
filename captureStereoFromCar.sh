#!/bin/bash
 
red=$(printf "\033[31m")
green=$(printf "\033[32m")
yellow=$(printf "\033[33m")
blue=$(printf "\033[34m")
magenta=$(printf "\033[35m")
cyan=$(printf "\033[36m")
white=$(printf "\033[37m")
normal=$(printf "\033[m")

normalChars=$(printf "\033[0m")
boldChars=$(printf "\033[1m")
underlinedChars=$(printf "\033[4m")
blinkingChars=$(printf "\033[5m") 


TIME_TO_RECORD="10000"
OUTPUT_FILE="stereoFromCar"



if (( $#<1 ))
then 
 echo "$red No argument supplied , please supply an argument with the output dataset $normal"
 exit 0
else
 OUTPUT_FILE=$1
fi
 

sudo modprobe usbcore usbfs_memory_mb=1000
 
#sudo nice -n -20 ionice -c 1 -n 0  -fps 20
./run_grabber.sh -module V4L2STEREO -from /dev/video2,/dev/video1 -resolution 752 416 -maxFrames $TIME_TO_RECORD -o $OUTPUT_FILE

exit 0
