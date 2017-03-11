#!/bin/bash

STYLE_MODEL="/home/ammar/Documents/3dParty/style-transfer"

STARTDIR=`pwd`
#Switch to this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"


rm -rf neuralVid
mkdir neuralVid

avconv -i $1 neuralVid/%07d.jpg 

cd "$DIR/neuralVid"
FILES_TO_CONVERT=`ls | grep .jpg`
for f in $FILES_TO_CONVERT
do 
 TARGETNAME=`basename $f .jpg` 
 convert $f -resize "640x480>^"  $STYLE_MODEL/images/content/job.jpg

 cd $STYLE_MODEL 
# cp $STYLE_MODEL/res2.jpg $STYLE_MODEL/res.jpg
  optirun python style.py -s images/style/starry_night.jpg -c images/content/job.jpg -m caffenet -g 0 -o res.jpg
 cd "$DIR/neuralVid"
 cp $STYLE_MODEL/res.jpg $f
 rm $STYLE_MODEL/res.jpg
done

THEDATETAG=`date +"%y-%m-%d_%H-%M-%S"` 
cd "$DIR/neuralVid"
avconv -i %07d.jpg -y -r 20 -threads 8 -b 30000k -s 640x480  ../outHD_$THEDATETAG.mp4 

exit 0
