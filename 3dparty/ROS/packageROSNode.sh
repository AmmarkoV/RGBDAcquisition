#!/bin/bash


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

red=$(printf "\033[31m")
green=$(printf "\033[32m")
yellow=$(printf "\033[33m")
blue=$(printf "\033[34m")
magenta=$(printf "\033[35m")
cyan=$(printf "\033[36m")
white=$(printf "\033[37m")
normal=$(printf "\033[m")


THEDATETAG=`date +"%y-%m-%d_%H-%M-%S"`

OURDISTRO=`lsb_release -a | grep "Distributor ID" | cut  -f2`
OURVER=`lsb_release -a | grep Release | cut  -f2`
echo "Our Distro is $OURDISTRO $OURVER"


NAME="RGBDAcquisitionROSNode$OURDISTRO$OURVER-$THEDATETAG"
 

tar cvfjh "$NAME.tar.bz2" rgbd_acquisition/
 

echo "$green Done.. $normal"
exit 0
