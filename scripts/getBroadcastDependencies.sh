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
 scripts/get_dependencies.sh

 ./make
 cd ..
else
 echo "Could not install AmmarServer"
fi

cd ..

ln -s acquisitionBroadcast/AmmarServer/public_html
cd acquisitionBroadcast
ln -s AmmarServer/public_html
cd ..




exit 0
