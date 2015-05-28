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
for (( contrast=1; contrast<=2; contrast++ ))
do   

 contrastFactor=$((contrast*20))  

 for (( i=0; i<=360; i++ ))
   do   
    convert classes/$f -brightness-contrast $contrastFactor -resize $DIMENSION -background white -gravity center -extent $DIMENSION -distort SRT  -$degs  -quality 99  dataset/$TARGETNAME/$i-$contrast.jpg
    degs=$((degs+1))  
    echo -n "."
   done 

done
 
done


exit 0
