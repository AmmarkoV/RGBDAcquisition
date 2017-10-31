#!/bin/bash


g++ `pkg-config --cflags opencv` templateMatching.cpp -o templateMatching `pkg-config --libs opencv`

exit 0
