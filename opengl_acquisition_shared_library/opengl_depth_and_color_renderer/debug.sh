#!/bin/bash
valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --track-origins=yes --num-callers=20 --track-fds=yes ./ModelDump suzanne.obj suzanne $@ 2>error.txt

#valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --track-origins=yes --num-callers=20 --track-fds=yes ./Renderer $@ 2>error.txt

exit $?
