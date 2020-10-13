#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

cd ..
cd ..
cd ..
cd ..

 
#./BVHTester --from Motions/05_01.bvh --selectJoints 0 23 hip eye.r eye.l abdomen chest neck head rshoulder relbow rhand lshoulder lelbow lhand rhip rknee rfoot lhip lknee lfoot toe1-2.r toe5-3.r toe1-2.l toe5-3.l --tuneIterations 130 4 130 0.01 5 30 20 > tuneIterations.dat



GNUPLOT_CMD="set terminal png size 1920,1400 font \"Helvetica,40\"; set output \"tuneIterations.png\"; set dgrid3d 190,190  qnorm 4; set isosample 160; set pm3d ; set palette defined (65 \"black\", 70 \"red\", 80 \"yellow\", 90 \"yellow\", 100 \"white\"); set view 60,45; set hidden3d; set xrange [1:25]; set yrange [9:60]; set zrange [30:90]; set style fill solid; set xlabel \"HCD Iterations\" rotate parallel; set ylabel \"Frames per second\" rotate parallel; set zlabel \"Mean average error in millimeters\" rotate parallel; splot \"tuneIterations.dat\" using 1:2:3 with lines title \"Tuning HCD Iterations Parameter\"; " 

gnuplot -e "$GNUPLOT_CMD"

timeout 5 gpicview tuneIterations.png

exit 0
