#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

cd scripts
./get-dependencies.sh
 
cd "$DIR"
scripts/refreshLinksTo3dParty.sh
scripts/createRedist.sh

scripts/checkEverything.sh

doxygen doc/doxyfile

exit 0
