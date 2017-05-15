#!/bin/bash


for f in $(find $1 -depth -type d); do  
       echo "Dir $f" 
       ./runRTPose.sh $f
done



exit 0
