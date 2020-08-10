#!/bin/bash


red=$(printf "\033[31m")
green=$(printf "\033[32m")
yellow=$(printf "\033[33m")
blue=$(printf "\033[34m")
magenta=$(printf "\033[35m")
cyan=$(printf "\033[36m")
white=$(printf "\033[37m")
normal=$(printf "\033[m")

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "Refreshing StringRecognizer build"
gcc main.c fastStringParser.c -o StringRecognizer

if [ -e StringRecognizer ]
then
#for item in TrajectoryPrimitives test ; do
item="TrajectoryPrimitives"
    echo -n "Generating $item header "
    ./StringRecognizer $item
    #gcc "$item.c" -o "$item-Scanner"

    cp "$item.c" ../
    cp "$item.h" ../

    rm "$item.c" "$item.h"

    echo "$green done $normal"
#done
else
 echo "$red Could not compile String recognizer..! $normal" 
fi
 
exit 0
