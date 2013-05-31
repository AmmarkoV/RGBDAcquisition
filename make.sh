#!/bin/bash

if [ -d grabber ]
then  
cd grabber
./make.sh
cd ..
fi

if [ -d acquisitionBroadcast ]
then  
cd acquisitionBroadcast
./make.sh
cd ..
fi

exit 0
