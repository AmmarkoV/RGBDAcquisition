#!/bin/bash


OBJNUM="16"


if [ "$#" -ne 3 ]; then
    echo "Illegal groundTruth fileA fileB"
    echo "groundTruth , fileA and fileB should look like itemID X Y Z<newline>"
    exit 2
fi

 
#Switch to this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"


count=0
while [ $count -le $OBJNUM ]
do
    ./comparePositionalData.sh $1 $2 "O$count"P0 
    ./comparePositionalData.sh $1 $3 "O$count"P0 
    (( count++ ))
done



exit 0
