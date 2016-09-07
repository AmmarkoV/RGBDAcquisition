#!/bin/bash
valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --track-origins=yes --num-callers=20 --track-fds=yes ./assimpTester ../../ScannedModels/Ammar3/AmmarRigged.dae Ammar.tri $@ 2>error.txt
 
valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --track-origins=yes --num-callers=20 --track-fds=yes ./assimpTester ../../ScannedModels/Elina5/ElinaRigged.dae Elina.tri $@ 2>error.txt

exit $?
