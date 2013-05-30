#!/bin/bash

DIRWITHINPUTPICS="v4l2StereoCalibrationTest"
 
 
for f in `ls "../frames/$DIRWITHINPUTPICS/" | grep 1_0.pnm`
do 
 echo "Processing $f "  
 cp ../frames/$DIRWITHINPUTPICS/$f cam0/
done

for f in `ls "../frames/$DIRWITHINPUTPICS/" | grep 1_1.pnm`
do 
 echo "Processing $f "  
 cp ../frames/$DIRWITHINPUTPICS/$f cam1/
done


