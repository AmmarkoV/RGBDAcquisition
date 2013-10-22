#!/bin/bash

X=7 

if (( $#<1 ))
then 
 echo "Will run as if you supplied 7 as an argument"
else
 X=$1
fi

echo "Clearing $X first images"

for (( i=$X-1; i>=0; i-- ))
do  
  XNUM=`printf %05u $X`
  INUM=`printf %05u $i`
  cp "colorFrame_0_$XNUM.pnm" "colorFrame_0_$INUM.pnm"
  cp "depthFrame_0_$XNUM.pnm" "depthFrame_0_$INUM.pnm"
done

exit 0
