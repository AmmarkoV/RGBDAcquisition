#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

cd ..
cd ..
cd ..
cd ..

 
#./BVHTester --from Motions/05_01.bvh --selectJoints 0 23 hip eye.r eye.l abdomen chest neck head rshoulder relbow rhand lshoulder lelbow lhand rhip rknee rfoot lhip lknee lfoot toe1-2.r toe5-3.r toe1-2.l toe5-3.l --tuneIterations 130 4 130 0.01 5 30 20 > tuneIterations.dat

# plot $map1 matrix using 1:2:3 with image,   $map1 matrix using 1:2:($3 == 0 ? "" : sprintf("%g",$3) ) with labels

GNUPLOT_CMD="set terminal png; set output \"tuneIterations.png\";  plot \"tuneIterations.dat\" matrix columnheaders using 1:2:3 with image"
gnuplot -e "$GNUPLOT_CMD"

exit 0
