#!/bin/bash
echo "This will resize all jpg images in INPUT_IMAGES directory to CONVERTED directory and rename them according to their input order..!"

mkdir $2

if [ -d "$1" ]; then
  echo "Found $1 directory"
else
  echo "Error : Could not find $1 directory" 
  exit 1
fi


if [ -d "$2" ]; then
  echo "Found $2 directory"
else
  echo "Error : Could not find $2 directory" 
  exit 1
fi

 

count=1
for i in $1/*; do  
  outname="$2/`printf colorFrame_0_%05d.jpg $count`"
  echo "Processing image $i output is $outname " ; 

  mv $i $outname;
  count=$((count + 1))
done 

echo "Done .."

exit 0

