#!/bin/bash
 
cd $1

mkdir ../$2
 

echo "Converting Color Files"

cd ims/color
FILES_TO_CONVERT=`ls | grep .png`
for f in $FILES_TO_CONVERT
do 
 TARGETNAME=`basename $f .png`
 TARGETNAME_TRIMMED=${TARGETNAME:1:${#TARGETNAME}}
 
 convert $f ../../../$2/colorFrame_0_$TARGETNAME_TRIMMED.pnm
done
cd ..
cd ..


echo "Converting Depth Files"

cd ims/depth
FILES_TO_CONVERT=`ls | grep .png`
for f in $FILES_TO_CONVERT
do 
 TARGETNAME=`basename $f .png`
 TARGETNAME_TRIMMED=${TARGETNAME:1:${#TARGETNAME}}
 ../../../../../tools/DepthImagesConverter/DepthImagesConverter "$f" "../../../$2/depthFrame_0_$TARGETNAME_TRIMMED.pnm"
done
cd ..
cd ..

cd ..
 
exit 0
