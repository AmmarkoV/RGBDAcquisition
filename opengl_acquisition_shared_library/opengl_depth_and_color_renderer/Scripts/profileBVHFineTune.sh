#!/bin/bash
THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$THISDIR"
cd ..

rm callgrind.out.*    

valgrind --tool=callgrind  ./BVHTester --from Motions/05_01.bvh --selectJoints 0 23 hip eye.r eye.l abdomen chest neck head rshoulder relbow rhand lshoulder lelbow lhand rhip rknee rfoot lhip lknee lfoot toe1-2.r toe5-3.r toe1-2.l toe5-3.l --testIK 80 4 130 0.001 5 100 1 $@


kcachegrind
exit 0
