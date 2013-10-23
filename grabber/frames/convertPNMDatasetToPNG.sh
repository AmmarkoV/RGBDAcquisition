#!/bin/bash
  
cd $1

mkdir ../$2

echo "Converting Color Files"

FILES_TO_CONVERT=`ls | grep color`
for f in $FILES_TO_CONVERT
do 
 TARGETNAME=`basename $f .pnm`
 convert $f ../$2/$TARGETNAME.png
done

FILES_TO_CONVERT=`ls | grep depth`
for f in $FILES_TO_CONVERT
do 
 TARGETNAME=`basename $f .pnm`
 ../../../tools/DepthImagesConverter/DepthImagesConverter "$f" "../$2/$TARGETNAME.png"
done

cd ..

exit 0
