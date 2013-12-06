#!/bin/bash

./get-dependencies.sh

cd 3dparty
./get_third_party_libs.sh

cd ..

scripts/refreshLinksTo3dParty.sh
scripts/createRedist.sh

scripts/checkEverything.sh

doxygen doc/doxyfile

exit 0
