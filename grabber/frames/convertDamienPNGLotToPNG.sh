#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
  
for D in `find $1 -type d -maxdepth 1 -mindepth 1`
do
 echo "Converting $D and outputting it to out/$D"
 #./convertDamienPNGToPNG.sh $D out/$D

 ./convertDamienJointDataset.sh $D/hyps.txt > $D/hyps.scene

 #we run the Neural network and get its 2D output
 ./runRTPose.sh $D 
 
 #we use 2d neural outputs to go to 3d points 
 cd ../../redist  
  rm neural3DOutput.scene
  ./TestBody --model AmmarH.tri --from $D -o neural3DOutput.scene
  cp neural3DOutput.scene frames/$D/neural3DOutput.scene
 cd $DIR 
 
 #we use 3d neural outputs to go to 3d fhbt model 
./NeuralNetworkJSONImporter $D/neural3DOutput.scene $D/neural3DOutput_to_FHBT.csv

done

 
 
exit 0
