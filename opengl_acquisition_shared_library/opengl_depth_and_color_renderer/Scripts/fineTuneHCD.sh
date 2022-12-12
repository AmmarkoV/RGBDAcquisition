#!/bin/bash
THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$THISDIR"
cd ..


PREVIOUS_FRAME="3"
CURRENT_FRAME="4"
TARGET_FRAME="40"
STEP_FRAME="20"
LR="0.01"
ITERATIONS="15"
EPOCHS="10"
LANGEVIN_DYNAMICS="0.3"
VERBOSITY="0"

for EPOCHS in `echo "5 10 15 20"`
do
 for ITERATIONS in `echo "3 5 10 15 20 25"`
 do
  for LR  in `echo "1 0.1 0.01 0.001"`
  do
   for LANGEVIN_DYNAMICS  in `echo "0.1 0.3 1 2"`
   do
      echo "IT: $ITERATIONS | EP: $EPOCHS | LR: $LR | LANGV: $LANGEVIN_DYNAMICS"
      ./BVHTester --from Motions/05_01.bvh --selectJoints 0 23 hip eye.r eye.l abdomen chest neck head rshoulder relbow rhand lshoulder lelbow lhand rhip rknee rfoot lhip lknee lfoot toe1-2.r toe5-3.r toe1-2.l toe5-3.l --testIK $PREVIOUS_FRAME $CURRENT_FRAME $TARGET_FRAME $STEP_FRAME $LR $ITERATIONS $EPOCHS 1 $LANGEVIN_DYNAMICS $VERBOSITY $@
   done
  done
 done 
done
#valgrind --tool=callgrind --dump-instr=yes --collect-jumps=yes ./BVHTester --from Motions/05_01.bvh --selectJoints 0 23 hip eye.r eye.l abdomen chest neck head rshoulder relbow rhand lshoulder lelbow lhand rhip rknee rfoot lhip lknee lfoot toe1-2.r toe5-3.r toe1-2.l toe5-3.l --testIK $PREVIOUS_FRAME $CURRENT_FRAME $TARGET_FRAME $STEP_FRAME $LR $ITERATIONS $EPOCHS 1 $LANGEVIN_DYNAMICS $@


#kcachegrind
exit 0
