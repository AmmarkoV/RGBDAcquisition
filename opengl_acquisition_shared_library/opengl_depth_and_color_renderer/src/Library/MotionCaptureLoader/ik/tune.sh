#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

cd ..
cd ..
cd ..
cd ..

 
#./BVHTester --from Motions/05_01.bvh --selectJoints 0 23 hip eye.r eye.l abdomen chest neck head rshoulder relbow rhand lshoulder lelbow lhand rhip rknee rfoot lhip lknee lfoot toe1-2.r toe5-3.r toe1-2.l toe5-3.l --tuneIterations 130 4 130 0.01 5 30 20 > tuneIterations.dat

# plot $map1 matrix using 1:2:3 with image,   $map1 matrix using 1:2:($3 == 0 ? "" : sprintf("%g",$3) ) with labels

GNUPLOT_CMD="set terminal png; set output \"tuneIterations.png\"; set xlabel \"Iterations\"; set ylabel \"Framerate\"; set zlabel \"Accuracy\";  set ticslevel 0; set samples 25, 25; set isosamples 50, 50; splot \"tuneIterations.dat\" matrix using 1:2:3 with pm3d title \"Iteration benchmark\""
gnuplot -e "$GNUPLOT_CMD"

exit 0
