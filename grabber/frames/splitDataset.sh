#!/bin/bash

FROMDATASET="$1"
TODATASET="$2" 
X=$3
TOTAL_FRAMES=100000

cd $FROMDATASET
mkdir ../$TODATASET

cp color.calib ../$TODATASET/color.calib
cp depth.calib ../$TODATASET/depth.calib
 
echo "Spliting Dataset $FROMDATASET from frame $X till its end to dataset $TODATASET "
echo "Please wait .. "
for (( i=$X; i<=$TOTAL_FRAMES; i++ ))
do   
  newIndx=$((i-X)) 
  XNUM=`printf %05u $i`
  INUM=`printf %05u $newIndx`
 
  if [ ! -f "colorFrame_0_$XNUM.pnm" ] 
    then
       if [ ! -f "depthFrame_0_$XNUM.pnm" ] 
         then
 
  if [ ! -f "colorFrame_0_$XNUM.jpg" ] 
    then
       if [ ! -f "depthFrame_0_$XNUM.png" ] 
         then
          echo " "
          echo "Finished Frames"
          exit 0
       fi
  fi
 
       fi
  fi

  mv "colorFrame_0_$XNUM.jpg" "../$TODATASET/colorFrame_0_$INUM.jpg"
  mv "depthFrame_0_$XNUM.png" "../$TODATASET/depthFrame_0_$INUM.png"


  mv "colorFrame_0_$XNUM.pnm" "../$TODATASET/colorFrame_0_$INUM.pnm"
  mv "depthFrame_0_$XNUM.pnm" "../$TODATASET/depthFrame_0_$INUM.pnm"
   
  echo -n "."
done

echo "Passed TOTAL_FRAMES (!) this is a bug! :S"

cd ..

exit 0
