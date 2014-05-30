#!/bin/bash
STARTDIR=`pwd`
BIN="GrabberSegment"
DIR="grabber_segment"
ORIGINALDIRBIN="../$DIR/$BIN"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd .. #This is positioned on tests/ dir
cd redist 

if [ -e $BIN ]
then
 ldd $BIN | grep not
 LD_LIBRARY_PATH=.:../3dparty/libfreenect/build/lib ./$BIN -module TEMPLATE -from floor -maxFrames 10 -to floorSegmented -autoplane 50 0 800 4000 -calibration frames/floor/color.calib -combine AND $@  &> ../error.txt
 gpicview frames/floorSegmented/colorFrame_0_00000.pnm
else
 if [ -e $ORIGINALDIRBIN ]
   then 
    echo "Could not find redist/$BIN , please consider running scripts/createRedist.sh"
    echo "Do you want to try that now ? " 
    echo
    echo -n " (Y/N)?"
    read answer
    if test "$answer" != "N" -a "$answer" != "n";
      then
       cd ..
       scripts/createRedist.sh
       cd redist
    fi
    echo "Please try to use $BIN again , and see if the problem is fixed" 
   else 
    echo "Could not find $BIN anywhere , please make sure you have compiled it successfully"
    echo "Do you want to try that automatically now ? " 
    echo
    echo -n " (Y/N)?"
    read answer
    if test "$answer" != "N" -a "$answer" != "n";
      then
       cd ..
        make
       cd redist
    fi
    echo "Please try to use $BIN again , and see if the problem is fixed" 
   fi  
fi 
 
cd "$STARTDIR"
exit 0
