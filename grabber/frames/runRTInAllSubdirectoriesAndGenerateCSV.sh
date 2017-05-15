#!/bin/bash
STARTDIR=`pwd` 
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"  


for f in $(find $1 -depth -type d); do  
       echo "Running Neural Network on Dataset.. $f" 
       ./runRTPose.sh $f 
done
 
echo "Compressing All Output.."
tar -czvf csvFiles.tar.gz `find  $1 -depth | grep '.csv'`



exit 0
