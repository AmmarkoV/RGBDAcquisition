#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"


./assimpTester --convert ../../ScannedModels/Ammar/AmmarRigged.dae Ammar.tri
./assimpTester --convert ../../ScannedModels/Elina/ElinaRigged.dae Elina.tri
./assimpTester --convert ../../ScannedModels/Dennis2/DennisRigged2.dae Dennis2.tri
./assimpTester --convert ../../ScannedModels/Aggeliki2/Aggeliki2Rigged.dae Aggeliki2.tri

exit 0
