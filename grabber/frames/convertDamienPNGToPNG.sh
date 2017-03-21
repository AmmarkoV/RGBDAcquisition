#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
 

cd $1
mkdir $DIR/$2
 

echo "Converting Color Files"
cd color
FILES_TO_CONVERT=`ls | grep .png`
for f in $FILES_TO_CONVERT
do 
 TARGETNAME=`basename $f .png`
 TARGETNAME_TRIMMED=${TARGETNAME:1:${#TARGETNAME}}
 
 cp $f  "$DIR/$2/colorFrame_0_$TARGETNAME_TRIMMED.png"
done
cd ..
cd ..


echo "Converting Depth Files"
cd depth
FILES_TO_CONVERT=`ls | grep .png`
for f in $FILES_TO_CONVERT
do 
 TARGETNAME=`basename $f .png`
 TARGETNAME_TRIMMED=${TARGETNAME:1:${#TARGETNAME}}
 cp "$f" "$DIR/$2/depthFrame_0_$TARGETNAME_TRIMMED.pnm"
done
cd ..
cd ..

cd ..
 
exit 0
