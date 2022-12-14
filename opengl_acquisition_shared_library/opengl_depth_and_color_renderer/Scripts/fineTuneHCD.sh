#!/bin/bash
THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$THISDIR"
cd ..

rm callgrind.out.* 
rm target_*.png
rm initial_*.png
rm solution_*.png
rm report_*.html
rm report.csv

DATASET="Motions/05_01.bvh"
DATASET="Motions/49_04.bvh"
DATASETA="/home/ammar/Documents/Programming/DNNTracker/DNNTracker/dataset/MotionCapture/121/121_12.bvh"
DATASETB="/home/ammar/Documents/Programming/DNNTracker/DNNTracker/dataset/MotionCapture/121/121_15.bvh"
DATASETC="/home/ammar/Documents/Programming/DNNTracker/DNNTracker/dataset/MotionCapture/122/122_27.bvh"

PREVIOUS_FRAME="3"
CURRENT_FRAME="4"
TARGET_FRAME="40"
STEP_FRAME="20"
LR="0.01"
ITERATIONS="15"
EPOCHS="10"
LANGEVIN_DYNAMICS="0.3"
VERBOSITY="0"

for EPOCHS in `echo "15"`
do
 for ITERATIONS in ` seq 20 1 25 | tr "," "." `
 do
  for LR  in ` seq 0.0018 0.0001 0.0026 | tr "," "." `
  do
   for LANGEVIN_DYNAMICS  in `seq 0.0 0.0005 0.005 | tr "," "."`
   do
      echo "IT: $ITERATIONS | EP: $EPOCHS | LR: $LR | LANGV: $LANGEVIN_DYNAMICS"
      ./BVHTester --from $DATASET --addfrom $DATASETA --addfrom $DATASETB --addfrom $DATASETC --selectJoints 0 23 hip eye.r eye.l abdomen chest neck head rshoulder relbow rhand lshoulder lelbow lhand rhip rknee rfoot lhip lknee lfoot toe1-2.r toe5-3.r toe1-2.l toe5-3.l --testIK $PREVIOUS_FRAME $CURRENT_FRAME $TARGET_FRAME $STEP_FRAME $LR $ITERATIONS $EPOCHS 1 $LANGEVIN_DYNAMICS $VERBOSITY $@
   done
  done
 done 
done


python3 Scripts/plotFineTune.py
#valgrind --tool=callgrind --dump-instr=yes --collect-jumps=yes ./BVHTester --from Motions/05_01.bvh --selectJoints 0 23 hip eye.r eye.l abdomen chest neck head rshoulder relbow rhand lshoulder lelbow lhand rhip rknee rfoot lhip lknee lfoot toe1-2.r toe5-3.r toe1-2.l toe5-3.l --testIK $PREVIOUS_FRAME $CURRENT_FRAME $TARGET_FRAME $STEP_FRAME $LR $ITERATIONS $EPOCHS 1 $LANGEVIN_DYNAMICS $@


#kcachegrind
exit 0
