#!/bin/bash

gcc -o gl3 glx3.c glx_test.c -lGL -lX11
./gl3

exit 0
