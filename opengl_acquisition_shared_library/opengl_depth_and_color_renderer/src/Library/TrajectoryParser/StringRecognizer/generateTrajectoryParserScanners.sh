#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "Refreshing StringRecognizer build"
gcc main.c fastStringParser.c -o StringRecognizer

if [ -e StringRecognizer ]
then
#for item in TrajectoryPrimitives test ; do

item="TrajectoryPrimitives"
    echo "Generating $item header"
    ./StringRecognizer $item
    #gcc "$item.c" -o "$item-Scanner"

    cp "$item.c" ../
    cp "$item.h" ../

    rm "$item.c" "$item.h"
#done
else
 echo "Could not compile String recognizer..!" 
fi
 
exit 0
