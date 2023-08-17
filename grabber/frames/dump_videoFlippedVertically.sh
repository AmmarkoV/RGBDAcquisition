#!/bin/bash 

#Simple dependency checker that will apt-get stuff if something is missing
# sudo apt-get install imagemagick
SYSTEM_DEPENDENCIES="ffmpeg"

for REQUIRED_PKG in $SYSTEM_DEPENDENCIES
do
PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep "install ok installed")
echo "Checking for $REQUIRED_PKG: $PKG_OK"
if [ "" = "$PKG_OK" ]; then

  echo "No $REQUIRED_PKG. Setting up $REQUIRED_PKG."

  #If this is uncommented then only packages that are missing will get prompted..
  #sudo apt-get --yes install $REQUIRED_PKG

  #if this is uncommented then if one package is missing then all missing packages are immediately installed..
  sudo apt-get install $SYSTEM_DEPENDENCIES  
  break
fi
done
#------------------------------------------------------------------------------


STARTDIR=`pwd`
#Switch to this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

DATASET=""

if (( $#<1 ))
then 
 echo "Please provide arguments first argument is dataset "
 exit 1
else
 DATASET=$1-data
fi
 
THEDATETAG=`date +"%y-%m-%d_%H-%M-%S"` 

mkdir $DATASET

ffmpeg -i $1  -r 30 -q:v 1 -vf "transpose=2,transpose=2"  $DATASET/colorFrame_0_%05d.jpg

cp $DATASET/colorFrame_0_00001.jpg $DATASET/colorFrame_0_00000.jpg

cd $DATASET
cd ..

cd $STARTDIR 
exit 0

