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
   
  foundColor=0
  if [ -f "colorFrame_0_$XNUM.png" ] 
    then
     mv "colorFrame_0_$XNUM.png" "../$TODATASET/colorFrame_0_$INUM.png"
     foundColor=1
    fi

  if [ -f "colorFrame_0_$XNUM.pnm" ] 
    then
     mv "colorFrame_0_$XNUM.pnm" "../$TODATASET/colorFrame_0_$INUM.pnm"
     foundColor=1
    fi

  if [ -f "colorFrame_0_$XNUM.jpg" ] 
    then
     mv "colorFrame_0_$XNUM.jpg" "../$TODATASET/colorFrame_0_$INUM.jpg"
     foundColor=1
    fi

  foundDepth=0
  if [ -f "depthFrame_0_$XNUM.png" ] 
    then
     mv "depthFrame_0_$XNUM.png" "../$TODATASET/depthFrame_0_$INUM.png"
     foundDepth=1
   fi 
  if [ -f "depthFrame_0_$XNUM.pnm" ] 
    then
     mv "depthFrame_0_$XNUM.pnm" "../$TODATASET/depthFrame_0_$INUM.pnm"
     foundDepth=1
   fi  
    
  if  [ "$foundColor" -eq "0" ]
    then
     if [ "$foundDepth" -eq "0" ]
      then
        echo "Finished @ $INUM frame"
        exit 0
      fi
  fi
   
  echo -n "."
done

echo "Passed TOTAL_FRAMES (!) this is a bug! :S"

cd ..

exit 0
