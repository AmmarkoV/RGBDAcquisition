#!/bin/bash
 
g++ `pkg-config --cflags --libs opencv`  sift.cpp -o sift

exit 0
