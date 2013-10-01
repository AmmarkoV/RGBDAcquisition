#!/bin/bash

#Switch to this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

cd ..


red=$(printf "\033[31m")
green=$(printf "\033[32m")
yellow=$(printf "\033[33m")
blue=$(printf "\033[34m")
magenta=$(printf "\033[35m")
cyan=$(printf "\033[36m")
white=$(printf "\033[37m")
normal=$(printf "\033[m")


OURDISTRO="`lsb_release -a | grep "Distributor ID" | cut  -f2`"
OURVER="`lsb_release -a | grep "Release" | cut  -f2`"

SEP="_"
OUROSSTR="$OURDISTRO$OURVER"
echo "Our os is $OUROSSTR"

THEDATETAG=`date +"%y-%m-%d_%H-%M-%S"`
tar cvfjh "RGBDAcquisition$OUROSSTR$SEP$THEDATETAG.tar.bz2" redist/
 

echo "$green Done.. $normal"




cd "$STARTDIR"

exit 0
