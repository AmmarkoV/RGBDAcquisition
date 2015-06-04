#!/bin/bash
  

INITIALSIZE=`du -hs $1 | cut -f1`

CONVNAME="tmp_under_conversion"
cd $1


mkdir ../$CONVNAME

cp color.calib ../$CONVNAME/color.calib
cp depth.calib ../$CONVNAME/depth.calib

echo "Converting Color Files"

FILES_TO_CONVERT=`ls | grep color`
for f in $FILES_TO_CONVERT
do 
 TARGETNAME=`basename $f .pnm`
 convert $f ../$CONVNAME/$TARGETNAME.jpg
done


echo "Converting Depth Files"

FILES_TO_CONVERT=`ls | grep depth`
for f in $FILES_TO_CONVERT
do 
 TARGETNAME=`basename $f .pnm`
 ../../../tools/DepthImagesConverter/DepthImagesConverter "$f" "../$CONVNAME/$TARGETNAME.png"
done

cd ..


rm -rf $1
mv $CONVNAME $1

FINALSIZE=`du -hs $1| cut -f1`

echo "Compressed from $INITIALSIZE to $FINALSIZE"



exit 0
