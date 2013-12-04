#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR" 


BINARIES_THAT_NEED_LIBS="`./binariesThatNeedLibs.sh`"



for f in $BINARIES_THAT_NEED_LIBS
           do  
             if [ -d ../$f/ ]
              then
               cd ../$f/ 
                    
               ../3dparty/link_to_libs.sh ../3dparty

               cd ../scripts/
             else
              echo "Could not create links for ../$f/ "
             fi
           done
 
cd "$STARTDIR" 

exit 0
