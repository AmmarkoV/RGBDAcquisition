#!/bin/bash
THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$THISDIR"
cd ..

rm callgrind.out.* 
rm target_*.png
rm initial_*.png
rm solution_*.png
rm report_*.html

MULTI_THREADING="--mt" #--mt 
NO_MULTI_THREADING=" " # or nothing

PREVIOUS_FRAME="3"
CURRENT_FRAME="4"
TARGET_FRAME="40"
STEP_FRAME="20"
STEP_FRAME="10"
LR="0.0020"
ITERATIONS="20"
EPOCHS="10"
LANGEVIN_DYNAMICS="0" #"0.002"
VERBOSITY="0"
LEARNING_RATE_DECAY="0.8"
MOMENTUM="0.42"

DATASETA="Motions/05_01.bvh"
DATASETB="Motions/49_04.bvh"


./BVHTester $NO_MULTI_THREADING --from $DATASETA --addfrom $DATASETB --selectJoints 0 23 hip eye.r eye.l abdomen chest neck head rshoulder relbow rhand lshoulder lelbow lhand rhip rknee rfoot lhip lknee lfoot toe1-2.r toe5-3.r toe1-2.l toe5-3.l --testIK $PREVIOUS_FRAME $CURRENT_FRAME $TARGET_FRAME $STEP_FRAME $LR $ITERATIONS $EPOCHS 1 $LANGEVIN_DYNAMICS $VERBOSITY $LEARNING_RATE_DECAY $MOMENTUM $@

#MAE in 2D Pixels went from 18.15 to 4.36 
#MAE in 3D mm went from 97.87 to 60.67 
#Computation time was 1052333 microseconds ( 0.95 fps )


./BVHTester $MULTI_THREADING --from $DATASETA --addfrom $DATASETB --selectJoints 0 23 hip eye.r eye.l abdomen chest neck head rshoulder relbow rhand lshoulder lelbow lhand rhip rknee rfoot lhip lknee lfoot toe1-2.r toe5-3.r toe1-2.l toe5-3.l --testIK $PREVIOUS_FRAME $CURRENT_FRAME $TARGET_FRAME $STEP_FRAME $LR $ITERATIONS $EPOCHS 1 $LANGEVIN_DYNAMICS $VERBOSITY $LEARNING_RATE_DECAY $MOMENTUM $@

#MAE in 2D Pixels went from 97.18 to 9.92 
#MAE in 3D mm went from 346.37 to 157.32 
#Computation time was 1052230 microseconds ( 0.95 fps )

exit 0
