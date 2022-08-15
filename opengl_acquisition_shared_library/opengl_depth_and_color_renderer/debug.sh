#!/bin/bash
#valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --track-origins=yes --num-callers=20 --track-fds=yes ./ModelDumpD Ammar.obj Ammar $@ 2>error.txt

#valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --track-origins=yes --num-callers=20 --track-fds=yes ./Renderer  --from Scenes/bvhTRI.conf $@ 2>error.txt

#valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --track-origins=yes --num-callers=20 --track-fds=yes ./BVHTester --testmultiple listOfFilesToCluster $@ 2>error.txt

#/home/ammar/Documents/3dParty/valgrind/bin/valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --track-origins=yes --num-callers=20 --track-fds=yes ./BVHTester --headrobot dataset/faces/neutral.bvh Head dataset/faces/list.txt dataset/faces/  $@ 2>error.txt



/home/ammar/Documents/3dParty/valgrind/bin/valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --track-origins=yes --num-callers=20 --track-fds=yes  ./BVHTester --printparams --haltonerror --from Motions/05_01.bvh --angleheatmap --filtergimballocks 4 --selectJoints 1 23 hip eye.r eye.l abdomen chest neck head rshoulder relbow rhand lshoulder lelbow lhand rhip rknee rfoot lhip lknee lfoot toe1-2.r toe5-3.r toe1-2.l toe5-3.l --hide2DLocationOfJoints 0 8 abdomen chest eye.r eye.l toe1-2.r toe5-3.r toe1-2.l toe5-3.l --perturbJointAngles 2 30.0 rshoulder lshoulder --perturbJointAngles 2 16.0 relbow lelbow --perturbJointAngles 2 10.0 abdomen chest --perturbJointAngles 2 30.0 rhip lhip --perturbJointAngles 4 10.0 lknee rknee lfoot rfoot --perturbJointAngles 2 10.0 abdomen chest --repeat 0 --sampleskip 2 --filterout 0 0 -130.0 0 90 0 1920 1080 570.7 570.3 6 rhand lhip 0 120 rhand rhip 0 120 rhand lhand 0 150 lhand rhip 0 120 lhand lhip 0 120 lhand rhand 0 150 --randomize2D 900 4500 -45 -179.999999 -45 45 180 45 --occlusions --csv debug/ body_all.csv 2d+3d+bvh $@ 2>error.txt


timeout 10 pluma error.txt

exit $?
