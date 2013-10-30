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
 convert $f ../$2/$TARGETNAME.png
done


echo "Converting Depth Files"

FILES_TO_CONVERT=`ls | grep depth`
for f in $FILES_TO_CONVERT
do 
 TARGETNAME=`basename $f .pnm`
 ../../../tools/DepthImagesConverter/DepthImagesConverter "$f" "../$2/$TARGETNAME.png"
done

cd ..

exit 0
