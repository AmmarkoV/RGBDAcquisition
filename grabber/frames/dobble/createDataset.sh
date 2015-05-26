#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

CLASS_LIST=`ls classes/`


for f in $CLASS_LIST
do 
 TARGETNAME=`basename $f .png`
 echo $TARGETNAME 
 mkdir -p dataset/$TARGETNAME
  

degs=0
for (( i=0; i<=35; i++ ))
do   
  convert classes/$f -gravity center -extent 110x110 -distort SRT  -$degs dataset/$TARGETNAME/$i.png
  degs=$((degs+10))  
  echo -n "."
done 
 
done


exit 0
