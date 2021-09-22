#!/bin/bash
#valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --track-origins=yes --num-callers=20 --track-fds=yes ./ModelDumpD Ammar.obj Ammar $@ 2>error.txt

#valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --track-origins=yes --num-callers=20 --track-fds=yes ./Renderer  --from Scenes/bvhTRI.conf $@ 2>error.txt

#valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --track-origins=yes --num-callers=20 --track-fds=yes ./BVHTester --testmultiple listOfFilesToCluster $@ 2>error.txt

/home/ammar/Documents/3dParty/valgrind/bin/valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --track-origins=yes --num-callers=20 --track-fds=yes ./BVHTester --headrobot dataset/faces/neutral.bvh Head dataset/faces/list.txt dataset/faces/  $@ 2>error.txt

exit $?
