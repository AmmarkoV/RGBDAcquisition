#!/bin/bash

cd acquisitionBroadcast

if [ -d AmmarServer ]
then
 rm -rf AmmarServer/
fi

git clone git://github.com/AmmarkoV/AmmarServer

if [ -d AmmarServer ]
then
 cd AmmarServer
 ./make
 cd ..
else
 echo "Could not install AmmarServer"
fi

cd ..


exit 0
