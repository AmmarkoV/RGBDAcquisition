#!/bin/bash
 

 
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR" 
cd ..

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


TIME_TO_RECORD="100000"
OUTPUT_FILE="stereoFromCar"



if (( $#<1 ))
then 
 echo "$red No argument supplied , please supply an argument with the output dataset $normal"
 exit 0
else
 OUTPUT_FILE=$1
fi



 
WIDTH="752"
HEIGHT="416"

#WIDTH="1280"
#HEIGHT="720"


WIDTH="320"
HEIGHT="240"

FPS="30"

sudo modprobe usbcore usbfs_memory_mb=1000
 
#Viewer
#./run_viewer.sh -module V4L2STEREO -from /dev/video1,/dev/video2 -resolution $WIDTH $HEIGHT -fps $FPS 


#Grabber
#sudo nice -n -20 ionice -c 1 -n 0 
./run_grabber.sh -module V4L2STEREO -from /dev/video0,/dev/video1 -resolution $WIDTH $HEIGHT -fps $FPS -maxFrames $TIME_TO_RECORD -o $OUTPUT_FILE

exit 0
