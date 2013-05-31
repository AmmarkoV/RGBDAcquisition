#!/bin/bash

if [ -d grabber ]
then  
cd grabber
./make.sh
cd ..
fi

if [ -d acquisitionBroadacst ]
then  
cd acquisitionBroadacst
./make.sh
cd ..
fi

exit 0
