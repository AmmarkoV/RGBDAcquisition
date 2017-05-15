#!/bin/bash
STARTDIR=`pwd` 
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"  


for f in $(find $1 -depth -type d); do  
       echo "Dir $f" 
      # ./runRTPose.sh $f 
done
 
echo "Compressing output"
tar -czvf csvFiles.tar.gz `find  $1 -depth | grep '.csv'`



exit 0
