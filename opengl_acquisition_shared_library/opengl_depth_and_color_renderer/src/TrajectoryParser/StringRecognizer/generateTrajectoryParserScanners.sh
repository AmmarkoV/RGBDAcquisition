#!/bin/bash


for item in TrajectoryPrimitives ; do
    echo "Generating $item header"
    ./StringRecognizer $item
    #gcc "$item.c" -o "$item-Scanner"

    cp "$item.c" ../
    cp "$item.h" ../

    rm "$item.c" "$item.h"
done

 
exit 0
