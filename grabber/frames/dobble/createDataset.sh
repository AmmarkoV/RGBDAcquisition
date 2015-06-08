#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

rm  -rf dataset/ 

CLASS_LIST=`ls classes/`

DIMENSION="32x32"

for f in $CLASS_LIST
do 
 TARGETNAME=`basename $f .png`
 echo $TARGETNAME 
 mkdir -p dataset/$TARGETNAME
  

degs=0
for (( contrast=1; contrast<=1; contrast++ ))
do   

 contrastFactor=$((contrast*20))  

 for (( i=0; i<=120; i++ ))
   do   
    convert classes/$f -brightness-contrast $contrastFactor -resize $DIMENSION -background white -gravity center -extent $DIMENSION -distort SRT  -$degs  -quality 99  dataset/$TARGETNAME/$i-$contrast.jpg
    degs=$((degs+3))  
    echo -n "."
   done 

done
 
done


/home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/caffe/build/examples/rgbd_acquisition_converter/rgbd_dataset_converter /home/ammar/Documents/Programming/FORTH/input_acquisition/grabbed_frames/dobble/dataset/



exit 0
