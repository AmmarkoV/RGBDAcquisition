#!/bin/bash

#https://github.com/DaveDubUK/bvhacker
#http://bvhacker.com/
#wget http://www.bvhacker.com/downloads/latest/bvhacker_1.8.zip

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

cd ~
HOMEDIR=`pwd`

PATHTOBVH="/.wine/drive_c/Program Files (x86)/bvhacker"
 

cd "$HOMEDIR/$PATHTOBVH"
wine bvhacker.exe


exit 0
