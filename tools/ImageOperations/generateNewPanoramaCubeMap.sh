#!/bin/bash


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

./imageopsutility $1 lastOutputCubeMap.jpg --envcube


exit 0
