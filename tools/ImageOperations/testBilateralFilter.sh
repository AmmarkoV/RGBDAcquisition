#!/bin/bash

STARTDIR=`pwd`  
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "$DIR" 

./imageopsutility ../../media/samples/test.jpg output.jpg

timeout 6 gpicview output.jpg

exit 0
