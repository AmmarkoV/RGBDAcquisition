#!/bin/bash 

#valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --track-origins=yes --num-callers=20 --track-fds=yes ./gl3MultiDiff -from PEOPLECAP/Ammar/color  $@ 2>error.txt

rm callgrind.out.*
valgrind --tool=callgrind  ./gl3MeshTransform --limit 4000 $@

kcachegrind

exit $?
