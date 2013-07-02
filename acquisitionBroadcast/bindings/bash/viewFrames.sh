#!/bin/bash

COUNTER=0

rgbFrames="http://139.91.185.49:8082/rgb.ppm"
depthFrames="http://139.91.185.49:8082/depth.ppm"



while [ 1 ] 
do
     echo "Loop $COUNTER"
     (time wget -qO- $rgbFrames > "rgb$COUNTER.pnm" ) 1> /dev/null 2>> times.txt& 
     (time wget -qO- $depthFrames > "depth$COUNTER.pnm" ) 1> /dev/null 2>> times.txt&
     sleep 0.50 
     COUNTER=$((COUNTER+1))
done

exit 0
