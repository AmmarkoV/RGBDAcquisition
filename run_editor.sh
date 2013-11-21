#!/bin/bash
STARTDIR=`pwd`
BIN="Editor"
DIR="editor"
ORIGINALDIRBIN="../$DIR/$BIN"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd redist 

if [ -e $BIN ]
then
 ldd $BIN | grep not
 LD_LIBRARY_PATH=. ./$BIN $@
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
       cd editor 
        make
       cd ..  
       cd redist
    fi
    echo "Please try to use $BIN again , and see if the problem is fixed" 
   fi  
fi 
 
cd "$STARTDIR"
exit 0
