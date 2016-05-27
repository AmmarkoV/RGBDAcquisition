#!/bin/bash
 
CDPATH="." 

mkdir -p obj/Debug

g++ -Wall -fexceptions -fPIC -I/usr/include/opencv -g  -c $CDPATH/affine.cpp -o obj/Debug/affine.o
g++ -Wall -fexceptions -fPIC -I/usr/include/opencv -g  -c $CDPATH/fundamental.cpp -o obj/Debug/fundamental.o
g++ -Wall -fexceptions -fPIC -I/usr/include/opencv -g  -c $CDPATH/homography.cpp -o obj/Debug/homography.o
g++ -Wall -fexceptions -fPIC -I/usr/include/opencv -g  -c $CDPATH/panorama.cpp -o obj/Debug/panorama.o
g++ -Wall -fexceptions -fPIC -I/usr/include/opencv -g  -c $CDPATH/reconstruction.cpp -o obj/Debug/reconstruction.o
g++ -Wall -fexceptions -fPIC -I/usr/include/opencv -g  -c $CDPATH/sift.cpp  -o obj/Debug/sift.o 
g++ -Wall -fexceptions -fPIC -I/usr/include/opencv -g  -c $CDPATH/stitcher.cpp -o obj/Debug/stitcher.o
g++ -Wall -fexceptions -fPIC -I/usr/include/opencv -g  -c $CDPATH/tools.cpp -o obj/Debug/tools.o

g++  -o panorama obj/Debug/affine.o obj/Debug/fundamental.o obj/Debug/homography.o obj/Debug/panorama.o obj/Debug/reconstruction.o obj/Debug/sift.o obj/Debug/stitcher.o obj/Debug/tools.o /usr/local/lib/libopencv_calib3d.so /usr/local/lib/libopencv_contrib.so /usr/local/lib/libopencv_core.so /usr/local/lib/libopencv_features2d.so /usr/local/lib/libopencv_flann.so /usr/local/lib/libopencv_gpu.so /usr/local/lib/libopencv_highgui.so /usr/local/lib/libopencv_imgproc.so /usr/local/lib/libopencv_legacy.so /usr/local/lib/libopencv_ml.so /usr/local/lib/libopencv_nonfree.so /usr/local/lib/libopencv_objdetect.so /usr/local/lib/libopencv_photo.so /usr/local/lib/libopencv_stitching.so /usr/local/lib/libopencv_superres.so /usr/local/lib/libopencv_ts.a /usr/local/lib/libopencv_video.so /usr/local/lib/libopencv_videostab.so -lrt -lpthread -lm -ldl 

 
 

exit 0
