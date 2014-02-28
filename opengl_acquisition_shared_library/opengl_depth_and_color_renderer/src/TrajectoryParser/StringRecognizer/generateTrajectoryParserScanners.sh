#!/bin/bash

#gcc main.c fastStringParser.c -o StringRecognizer

#for item in TrajectoryPrimitives test ; do

item="TrajectoryPrimitives"
    echo "Generating $item header"
    ./StringRecognizer $item
    #gcc "$item.c" -o "$item-Scanner"

    cp "$item.c" ../
    cp "$item.h" ../

    rm "$item.c" "$item.h"
#done

 
exit 0
