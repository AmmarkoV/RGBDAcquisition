#!/bin/bash
 

OPEN_CV_LIBS="/home/ammar/Documents/3dParty/opencv-2.4.9/build/lib/libopencv_calib3d.so -lopencv_calib3d /home/ammar/Documents/3dParty/opencv-2.4.9/build/lib/libopencv_contrib.so -lopencv_contrib /home/ammar/Documents/3dParty/opencv-2.4.9/build/lib/libopencv_core.so -lopencv_core /home/ammar/Documents/3dParty/opencv-2.4.9/build/lib/libopencv_features2d.so -lopencv_features2d /home/ammar/Documents/3dParty/opencv-2.4.9/build/lib/libopencv_flann.so -lopencv_flann /home/ammar/Documents/3dParty/opencv-2.4.9/build/lib/libopencv_gpu.so -lopencv_gpu /home/ammar/Documents/3dParty/opencv-2.4.9/build/lib/libopencv_highgui.so -lopencv_highgui /home/ammar/Documents/3dParty/opencv-2.4.9/build/lib/libopencv_imgproc.so -lopencv_imgproc /home/ammar/Documents/3dParty/opencv-2.4.9/build/lib/libopencv_legacy.so -lopencv_legacy /home/ammar/Documents/3dParty/opencv-2.4.9/build/lib/libopencv_ml.so -lopencv_ml /home/ammar/Documents/3dParty/opencv-2.4.9/build/lib/libopencv_objdetect.so -lopencv_objdetect /home/ammar/Documents/3dParty/opencv-2.4.9/build/lib/libopencv_ocl.so -lopencv_ocl /home/ammar/Documents/3dParty/opencv-2.4.9/build/lib/libopencv_photo.so -lopencv_photo /home/ammar/Documents/3dParty/opencv-2.4.9/build/lib/libopencv_stitching.so -lopencv_stitching /home/ammar/Documents/3dParty/opencv-2.4.9/build/lib/libopencv_superres.so -lopencv_superres /home/ammar/Documents/3dParty/opencv-2.4.9/build/lib/libopencv_video.so -lopencv_video /home/ammar/Documents/3dParty/opencv-2.4.9/build/lib/libopencv_videostab.so -lopencv_videostab /home/ammar/Documents/3dParty/opencv-2.4.9/build/lib/libopencv_nonfree.so.2.4.9"

OPENCV_INCLUDES="-I/home/ammar/Documents/3dParty/opencv-2.4.9/include"

g++ `pkg-config --cflags --libs opencv` -L. sift.cpp -o sift


#g++ $OPENCV_INCLUDES $OPEN_CV_LIBS  sift.cpp -o sift

exit 0
