#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR" 

mv $1 $1-Uncompressed
./convertPNMDatasetToJPGPNG.sh $1-Uncompressed $1

exit 0
