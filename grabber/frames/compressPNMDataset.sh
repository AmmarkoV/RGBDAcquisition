#!/bin/bash
  
 

INITIALSIZE=`du -hs $1 | cut -f1`

CONVNAME="tmp_under_conversion"
cd $1



COLORFILENUM=`ls -al | grep color | grep .pnm |  wc -l`
DEPTHFILENUM=`ls -al | grep depth | grep .pnm |  wc -l`


if [ $COLORFILENUM -eq 0 ]; then
 echo "No color pnm files found , $1 is not a valid PNM dataset for compression..!"
 exit 0
fi
if [ $DEPTHFILENUM -eq 0 ]; then
 echo "No depth pnm files found , $1 is not a valid PNM dataset for compression..!"
 exit 0
fi

echo "Looks $1 is an uncompressed dataset ($COLORFILENUM pnm color files , $DEPTHFILENUM pnm depth files ) will try to compress it now "

exit 0
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

FINALSIZE=`du -hs $CONVNAME| cut -f1`
  echo
  echo   
echo "Compressed from $INITIALSIZE to $FINALSIZE"


echo "Overwrite old dataset ?"
  echo
  echo   
  echo ""
  echo
  echo -n "            (Y/N)?"
  read answer
  if test "$answer" != "Y" -a "$answer" != "y";
  then 
   echo "Naming the compressed dataset $1-Compressed.." 
   mv $CONVNAME "$1-Compressed" 
else
   echo "Overwriting.." 
   rm -rf $1
   mv $CONVNAME $1 
fi

 




exit 0
