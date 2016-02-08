#!/bin/bash
  

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR" 

cd $1

mkdir ../$2

cp *.calib ../$2/
cp *.sh ../$2/

echo "Converting Color Files"

FILES_TO_CONVERT=`ls | grep color`
for f in $FILES_TO_CONVERT
do 
 TARGETNAME=`basename $f .pnm`
 convert -quality 95 $f ../$2/$TARGETNAME.jpg
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
