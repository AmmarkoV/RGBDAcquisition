#!/bin/bash
THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$THISDIR"
cd ..

rm callgrind.out.* 
rm target_*.png
rm initial_*.png
rm solution_*.png
rm report_*.html

PREVIOUS_FRAME="3"
CURRENT_FRAME="4"
TARGET_FRAME="40"
STEP_FRAME="20"
STEP_FRAME="10"
LR="0.0020"
ITERATIONS="20"
EPOCHS="10"
LANGEVIN_DYNAMICS="0.002"
VERBOSITY="0"

DATASETA="Motions/05_01.bvh"
DATASETB="Motions/49_04.bvh"

valgrind --tool=callgrind --dump-instr=yes --collect-jumps=yes ./BVHTester --from $DATASETA --addfrom $DATASETB --selectJoints 0 23 hip eye.r eye.l abdomen chest neck head rshoulder relbow rhand lshoulder lelbow lhand rhip rknee rfoot lhip lknee lfoot toe1-2.r toe5-3.r toe1-2.l toe5-3.l --testIK $PREVIOUS_FRAME $CURRENT_FRAME $TARGET_FRAME $STEP_FRAME $LR $ITERATIONS $EPOCHS 1 $LANGEVIN_DYNAMICS $VERBOSITY $@


kcachegrind
exit 0
