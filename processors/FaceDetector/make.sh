#!/bin/bash

g++ faceDetector.cpp -o faceDetector `pkg-config --cflags --libs opencv`

exit 0
