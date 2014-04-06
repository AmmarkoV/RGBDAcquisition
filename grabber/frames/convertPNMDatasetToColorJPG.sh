#!/bin/bash
  
cd $1

mkdir ../$2

cp color.calib ../$2/color.calib
cp depth.calib ../$2/depth.calib

echo "Converting Color Files"

FILES_TO_CONVERT=`ls | grep color`
for f in $FILES_TO_CONVERT
do 
 TARGETNAME=`basename $f .pnm`
 convert $f ../$2/$TARGETNAME.jpg
done

 
cd ..

exit 0
