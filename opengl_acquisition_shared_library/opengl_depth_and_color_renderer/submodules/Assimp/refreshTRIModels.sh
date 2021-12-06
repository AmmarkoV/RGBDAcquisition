#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"


MHX2MODEL="makehuman2ThroughMHX2.dae"
PREFIX="Test_"
MHX2MODEL="makehuman2ThroughMHX2Default.dae"
PREFIX="Testexp_"

#Export Collada:
#Select SL + OpenSim rigged from presets

./assimpTester --merge axis.dae axis.tri

#valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --track-origins=yes --num-callers=20 --track-fds=yes ./assimpTester --merge axis.obj axis.tri $@ 2>error.txt

#Mesh import from MHX2 intermediate format..
./assimpTester --mesh 0 --convert $MHX2MODEL makehuman.tri --removeprefix $PREFIX --packtexture young_lightskinned_female_diffuse3.png  --applytexture young_lightskinned_female_diffuse3.png --droptexturealpha
./assimpTester --mesh 1 --convert $MHX2MODEL hair.tri --removeprefix $PREFIX --packtexture braid01_diffuse_mahogany.png --applytexture braid01_diffuse_mahogany.png --droptexturealpha
./assimpTester --mesh 2 --convert $MHX2MODEL eyes.tri --removeprefix $PREFIX --packtexture brown_eye.png --applytexture brown_eye.png --droptexturealpha
./assimpTester --mesh 3 --convert $MHX2MODEL eyebrows.tri --removeprefix $PREFIX --packtexture eyebrow001.png --applytexture eyebrow001.png --droptexturealpha
./assimpTester --mesh 4 --convert $MHX2MODEL eyelashes.tri --removeprefix $PREFIX --packtexture eyelashes01.png --applytexture eyelashes01.png --droptexturealpha
./assimpTester --mesh 5 --convert $MHX2MODEL teeth.tri --removeprefix $PREFIX --packtexture teeth.png --applytexture teeth.png --droptexturealpha
./assimpTester --mesh 6 --convert $MHX2MODEL tongue.tri --removeprefix $PREFIX --packtexture tongue01_diffuse.png --applytexture tongue01_diffuse.png --droptexturealpha


exit 0

scp -P 2222 axis.tri  makehuman.tri hair.tri eyes.tri eyebrows.tri eyelashes.tri teeth.tri tongue.tri ammar@ammar.gr:/home/ammar/public_html/mocapnet/mnet4/




#Stable mesh
./assimpTester --mesh 0 --convert makehuman2.dae makehuman.tri --applytexture young_lightskinned_female_diffuse3.png 
./assimpTester --mesh 1 --convert makehuman2.dae hair.tri --applytexture braid01_diffuse_mahogany.png
./assimpTester --mesh 2 --convert makehuman2.dae eyes.tri --applytexture brown_eye.png

exit 0


./assimpTester --convert eyes.dae eyes.tri #--applytexture brown_eye.png

./assimpTester --convert makehumanexports/newmodel.dae makehuman.tri --applytexture young_lightskinned_female_diffuse3.png 

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

exit 0
