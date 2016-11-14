#!/bin/bash

echo "Convert Hands.tri"
valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --track-origins=yes --num-callers=20 --track-fds=yes ./assimpTester --convert ../../Models/hands_v8.dae hands_v8.tri $@ 2>errorH.txt


echo "Convert Ammar.tri"
valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --track-origins=yes --num-callers=20 --track-fds=yes ./assimpTester --convert ../../ScannedModels/Ammar3/AmmarRigged.dae Ammar.tri $@ 2>error1.txt
 
echo "Convert Elina.tri"
valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --track-origins=yes --num-callers=20 --track-fds=yes ./assimpTester --convert ../../ScannedModels/Elina5/ElinaRigged.dae Elina.tri $@ 2>error2.txt

exit $?
