#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR" 

cd ..
cd ..
 
./run_grabber.sh -module OPENNI2 -to /dev/null -maxFrames 5000  2> out.txt

cat out.txt | grep BBoxSize | cut -d ' ' -f 2 > openni2_acquisition_shared_library/scripts/bboxx.txt
cat out.txt | grep BBoxSize | cut -d ' ' -f 3 > openni2_acquisition_shared_library/scripts/bboxy.txt
cat out.txt | grep BBoxSize | cut -d ' ' -f 4 > openni2_acquisition_shared_library/scripts/bboxz.txt

cd openni2_acquisition_shared_library/scripts
./generateGraphs.sh


exit 0
