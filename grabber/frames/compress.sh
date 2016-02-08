#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR" 

mv $1 $1-Uncompressed
./convertPNMDatasetToJPGPNG.sh $1 $2

exit 0
