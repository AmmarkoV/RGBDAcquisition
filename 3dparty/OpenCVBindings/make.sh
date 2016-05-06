#!/bin/bash

g++  `pkg-config --cflags opencv` `pkg-config --libs opencv` -lopencv_features2d sift.cpp -o sift

exit 0
