#!/bin/bash



INPUTFILE="pnm"

if (( $#<1 ))
then 
 echo "Please provide arguments first argument is dataset ,  second is file format ( i.e. boxNew jpg ) "
 exit 1
else
 INPUTFILE=$1 
fi
 

convert $INPUTFILE -crop 720x416+0+0 part1.jpg
convert $INPUTFILE -crop 720x416+771+0 part2.jpg

convert -delay 10  -size 720x416 -page +0+0  part1.jpg -page +0+0  part2.jpg -loop 0  animation.gif

exit 0
