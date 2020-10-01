#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

cd ..
cd ..
cd ..
cd ..


rm ikCachegrind.out*

valgrind  --separate-threads=yes --tool=cachegrind  --tool=callgrind --dump-instr=yes --collect-jumps=yes --callgrind-out-file=ikCachegrind.out ./BVHTester --from Motions/05_01.bvh --selectJoints 0 23 hip eye.r eye.l abdomen chest neck head rshoulder relbow rhand lshoulder lelbow lhand rhip rknee rfoot lhip lknee lfoot toe1-2.r toe5-3.r toe1-2.l toe5-3.l --mt --testIK 130 4 130 0.01 5 30 20

 
kcachegrind ikCachegrind.out*

exit 0
