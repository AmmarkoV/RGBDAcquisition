#!/bin/bash
cd frames
 for filename in *  #loop through all the files
 do
 fname=`basename $filename`
 convert $fname $fname".jpg" # replace .pnm with .jpg in the filename
 done
cd ..
exit 0
