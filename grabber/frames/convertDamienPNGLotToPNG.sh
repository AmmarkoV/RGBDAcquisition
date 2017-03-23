#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
  
for D in `find $1 -type d -maxdepth 1 -mindepth 1`
do
 echo "Converting $D and outputting it to out/$D"
 ./convertDamienPNGToPNG.sh $D out/$D
done

 
 
exit 0
