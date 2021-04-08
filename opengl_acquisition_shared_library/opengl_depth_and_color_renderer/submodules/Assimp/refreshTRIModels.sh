#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"


./assimpTester --convert ../../ScannedModels/Ammar/AmmarRigged.dae AmmarO.tri
./assimpTester --convert ../../ScannedModels/Elina/ElinaRigged.dae ElinaO.tri
./assimpTester --convert ../../ScannedModels/Dennis2/DennisRigged2.dae Dennis2O.tri
./assimpTester --convert ../../ScannedModels/Aggeliki2/Aggeliki2Rigged.dae Aggeliki2O.tri


./assimpTester --convert AmmarRiggedH.dae AmmarH.tri
./assimpTester --convert AmmarRiggedM.dae AmmarM.tri
./assimpTester --convert AmmarRiggedS.dae AmmarS.tri


./assimpTester --convert Ammar_1k.dae Ammar_1k.tri
./assimpTester --convert Ammar_7k.dae Ammar_7k.tri
./assimpTester --convert Ammar_18k.dae Ammar_18k.tri 

./assimpTester --convert ElinaRiggedH.dae ElinaH.tri
./assimpTester --convert ElinaRiggedS.dae ElinaS.tri


ln -s AmmarH.tri Ammar.tri
ln -s ElinaH.tri Elina.tri

#./assimpTester --convert makehumanexports/newmodel.dae makehuman.tri --paint 123 123 123
./assimpTester --convert makehumanexports/newmodel.dae makehuman.tri --applytexture young_lightskinned_female_diffuse3.png 

exit 0
