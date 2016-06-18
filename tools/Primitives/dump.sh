#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

rostopic echo /joints2D3D  > dump.skel 

exit 0
