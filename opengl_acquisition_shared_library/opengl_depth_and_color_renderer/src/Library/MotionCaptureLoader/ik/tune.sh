#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

cd ..
cd ..
cd ..
cd ..

 
#./BVHTester --from Motions/05_01.bvh --selectJoints 0 23 hip eye.r eye.l abdomen chest neck head rshoulder relbow rhand lshoulder lelbow lhand rhip rknee rfoot lhip lknee lfoot toe1-2.r toe5-3.r toe1-2.l toe5-3.l --tuneIterations 130 4 130 0.01 5 30 20 > tuneIterations.dat


#set view 45,45; set view map; set yrange [9:90]; set zrange [30:90];
# set dgrid3d 25,25 qnorm 4;
#splot \"tuneIterations.dat\" using 1:3:2 with points pointsize 3 pointtype 7; 
GNUPLOT_CMD="set terminal png size 1920,1400 font \"Helvetica,40\"; set output \"tuneIterations.png\";  set isosample 160; set pm3d ; set palette defined (65 \"black\", 70 \"red\", 80 \"yellow\", 90 \"yellow\", 100 \"white\"); set view map; set hidden3d; set xrange [1:25]; set style fill solid; set xlabel \"HCD Number of iterations performed\" rotate parallel; set ylabel \"Frames per second\" rotate parallel; set zlabel \"Mean average error in millimeters\" rotate parallel; set title \"HCD tuning iteration hyperparameter\"; set multiplot; splot \"tuneIterations.dat\" using 1:3:2 with lines palette title \" \"; splot \"tuneIterations.dat\" using 1:3:2 with points palette pointsize 5 pointtype 7 title \" \";  " 

gnuplot -e "$GNUPLOT_CMD"

timeout 5 gpicview tuneIterations.png

exit 0
