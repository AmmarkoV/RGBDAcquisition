#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

if [ ! -f /usr/lib/libOpenCL.so ] 
then 
 sudo ln -s /usr/local/cuda/lib64/libOpenCL.so /usr/lib/libOpenCL.so
fi

gcc openCLRenderer.c openCLTools.c -o bvhtransform -lOpenCL


exit 0
