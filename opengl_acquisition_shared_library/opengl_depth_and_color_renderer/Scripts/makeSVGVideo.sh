#!/bin/bash 
 


ffmpeg -i %06d.svg -y -r 25 -threads 8 -b 30000k -s 640x480  ./visualization.mp4 
exit 0



#if we want to perform rasterization using inscape you can use following code..

THEDATETAG=`date +"%y-%m-%d_%H-%M-%S"` 
pwd 
mkdir png
FILES=./*

COUNTER=0 
for f in $FILES
do
 FILENAMENOEXT=`
     FULL_FILENAME=$f 
     FILENAME=${FULL_FILENAME##*/}
     echo ${FILENAME%%.*}`
 
    ISITAFRAME=`echo $f | grep "svg"` 
     if [ -n "$ISITAFRAME" ] 
      then
       file2=${FILENAMENOEXT/"svg"/"particleScore_"}

       echo "converting $FILENAMENOEXT.svg to  png/$file2.png" 
       #convert $FILENAMENOEXT.svg  png/$file2.png
       inkscape -z -e png/$file2.png -w 640 -h 480 $FILENAMENOEXT.svg 
       
       
       COUNTER=$[$COUNTER +1] 
     fi 
done  

ffmpeg -i png/%06d.png -y -r 20 -threads 8 -b 30000k -s 640x480  ./visualization.mp4 

rm -rf png/

exit 0

