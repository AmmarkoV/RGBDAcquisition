#!/bin/bash

sudo mkfifo /dev/myGPS

while [ -e $1 ]
do
 cat $1 > /dev/myGPS
 sleep 1
done 

exit 0
