#!/bin/bash

sudo apt-get install build-essential cmake

cd 3dparty
./get_third_party_libs.sh
cd ..

scripts/getBroadcastDependencies.sh

exit 0
