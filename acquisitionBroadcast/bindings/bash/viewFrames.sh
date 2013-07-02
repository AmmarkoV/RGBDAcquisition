#!/bin/bash

COUNTER=0

snapFrames="http://139.91.185.49:8080/control.html?snap=1"
rgbFrames="http://139.91.185.49:8080/rgb.ppm"
depthFrames="http://139.91.185.49:8080/depth.ppm"



while [ 1 ] 
do
     echo "Loop $COUNTER"
     wget -qO- $snapFrames >  /dev/null 
     (time wget -qO- $rgbFrames > "rgb$COUNTER.pnm" ) 1> /dev/null 2>> times.txt& 
     (time wget -qO- $depthFrames > "depth$COUNTER.pnm" ) 1> /dev/null 2>> times.txt&
     #sleep 0.10 
     COUNTER=$((COUNTER+1))
done

exit 0
