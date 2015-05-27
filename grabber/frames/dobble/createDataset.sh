#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

CLASS_LIST=`ls classes/`

DIMENSION="32x32"

for f in $CLASS_LIST
do 
 TARGETNAME=`basename $f .png`
 echo $TARGETNAME 
 mkdir -p dataset/$TARGETNAME
  

degs=0
for (( i=0; i<=35; i++ ))
do   
  convert classes/$f -resize $DIMENSION -background white -gravity center -extent $DIMENSION -distort SRT  -$degs  -quality 99  dataset/$TARGETNAME/$i.jpg
  degs=$((degs+10))  
  echo -n "."
done 
 
done


exit 0
