#!/bin/bash

sudo mkfifo /dev/myGPSA

while [ -e $1 ]
do
 cat $1 > /dev/myGPSA
 sleep 1
done 

exit 0
