#!/bin/bash

STYLE_MODEL="/home/ammar/Documents/3dParty/style-transfer"
STYLE_FILE="$STYLE_MODEL/images/style/work$2.jpg"
IMAGE_FOLDER=$1_using_$2_Video

cp $2 $STYLE_FILE


STARTDIR=`pwd`
#Switch to this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"


rm -rf $IMAGE_FOLDER
mkdir $IMAGE_FOLDER

avconv -i $1 $IMAGE_FOLDER/%07d.jpg 

count=1

cd "$DIR/$IMAGE_FOLDER"
FILES_TO_CONVERT=`ls | grep .jpg`
for f in $FILES_TO_CONVERT
do 
 TARGETNAME=`basename $f .jpg` 
 convert $f -resize "640x480>^"  $STYLE_MODEL/images/content/job.jpg

 cd $STYLE_MODEL 
 python style.py -s $STYLE_FILE -c images/content/job.jpg -m caffenet -g 0 -o res.jpg
 cd "$DIR/$IMAGE_FOLDER"
 cp $STYLE_MODEL/res.jpg $f
 rm $STYLE_MODEL/res.jpg


  count=$((count + 1))
  echo "Frame $count" > $DIR/$IMAGE_FOLDER/progress.txt
done

THEDATETAG=`date +"%y-%m-%d_%H-%M-%S"` 
cd "$DIR/$IMAGE_FOLDER"
avconv -i %07d.jpg -y -r 20 -threads 8 -b 30000k -s 640x480  ../outHD_$1_using_$2_$THEDATETAG.mp4 

exit 0
