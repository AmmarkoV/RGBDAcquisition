#!/bin/bash
 
CDPATH="." 

mkdir -p obj/Debug

g++ -Wall -fexceptions -fPIC -I/usr/include/opencv -g  -c $CDPATH/affine.cpp -o obj/Debug/affine.o
g++ -Wall -fexceptions -fPIC -I/usr/include/opencv -g  -c $CDPATH/homography.cpp -o obj/Debug/homography.o
g++ -Wall -fexceptions -fPIC -I/usr/include/opencv -g  -c $CDPATH/panorama.cpp -o obj/Debug/panorama.o
g++ -Wall -fexceptions -fPIC -I/usr/include/opencv -g  -c $CDPATH/sift.cpp  -o obj/Debug/sift.o 
g++ -Wall -fexceptions -fPIC -I/usr/include/opencv -g  -c $CDPATH/stitcher.cpp -o obj/Debug/stitcher.o
g++ -Wall -fexceptions -fPIC -I/usr/include/opencv -g  -c $CDPATH/tools.cpp -o obj/Debug/tools.o

g++  -o panorama obj/Debug/affine.o obj/Debug/homography.o obj/Debug/panorama.o obj/Debug/sift.o obj/Debug/stitcher.o obj/Debug/tools.o  /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so /usr/lib/x86_64-linux-gnu/libopencv_contrib.so /usr/lib/x86_64-linux-gnu/libopencv_core.so /usr/lib/x86_64-linux-gnu/libopencv_features2d.so /usr/lib/x86_64-linux-gnu/libopencv_flann.so /usr/lib/x86_64-linux-gnu/libopencv_gpu.so /usr/lib/x86_64-linux-gnu/libopencv_highgui.so /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so /usr/lib/x86_64-linux-gnu/libopencv_legacy.so /usr/lib/x86_64-linux-gnu/libopencv_ml.so /usr/lib/x86_64-linux-gnu/libopencv_nonfree.so /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so /usr/lib/x86_64-linux-gnu/libopencv_ocl.so /usr/lib/x86_64-linux-gnu/libopencv_photo.so /usr/lib/x86_64-linux-gnu/libopencv_stitching.so /usr/lib/x86_64-linux-gnu/libopencv_superres.so /usr/lib/x86_64-linux-gnu/libopencv_ts.so /usr/lib/x86_64-linux-gnu/libopencv_video.so /usr/lib/x86_64-linux-gnu/libopencv_videostab.so -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab   






exit 0
