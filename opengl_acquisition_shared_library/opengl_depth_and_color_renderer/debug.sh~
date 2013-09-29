#!/bin/bash
valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=20 --track-fds=yes ./Renderer -from dragon.conf -photo 1 1 2 3 -size 1048 1048 2>error.txt
