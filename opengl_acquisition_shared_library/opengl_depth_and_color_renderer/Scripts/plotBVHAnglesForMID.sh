#!/bin/bash

BVHFILE="$1" 

NUMBEROFCHANNELS=`grep -A1 -P '^Frame Time: 0.04166667$' $BVHFILE | wc -w`
echo "BVH File $BVHFILE has $NUMBEROFCHANNELS channels"



for i in $(seq 1 $NUMBEROFCHANNELS);
do
  echo "plotting BVH $BVHFILE / MotionID $i"
  grep -A10000 -P '^Frame Time: 0.04166667$' $BVHFILE | cut -d ' ' -f $i  > temp.csv
  python3 Scripts/plot1D.py --from temp.csv --to "out$i.png"
done
rm temp.csv


#grep -A10000 -P '^Frame Time: 0.04166667$' $BVHFILE | cut -d ' ' -f $BVHMOTIONID  > temp.csv
#python3 Scripts/plot1D.py --from temp.csv 
#rm temp.csv

exit 0
