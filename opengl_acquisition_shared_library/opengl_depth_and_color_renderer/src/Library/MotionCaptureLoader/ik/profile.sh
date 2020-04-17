#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

cd ..
cd ..
cd ..
cd ..


rm ikCachegrind.log

valgrind --tool=cachegrind  --tool=callgrind --dump-instr=yes --collect-jumps=yes --callgrind-out-file=ikCachegrind.log ./BVHTester --from Motions/05_01.bvh --selectJoints 0 23 hip eye.r eye.l abdomen chest neck head rshoulder relbow rhand lshoulder lelbow lhand rhip rknee rfoot lhip lknee lfoot toe1-2.r toe5-3.r toe1-2.l toe5-3.l --testIK 130 4 130 0.001 5 100

 
kcachegrind ikCachegrind.log

exit 0
