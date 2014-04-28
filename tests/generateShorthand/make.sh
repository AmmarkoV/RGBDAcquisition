#!/bin/bash

gcc  generate_shorthand.cpp -o generate

./generate 2>  out.txt

echo " Got "
cat out.txt | wc -l 
echo " results in out.txt"

exit 0
