#!/bin/bash
 
 
 ./run_viewer.sh -module OPENNI2 -from 0 -processor ../processors/FaceDetector/libFaceDetector.so  FaceDetector  $@
 




exit 0
