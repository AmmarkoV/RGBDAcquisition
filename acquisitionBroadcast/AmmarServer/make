#!/bin/bash
 
red=$(printf "\033[31m")
green=$(printf "\033[32m") 
normal=$(printf "\033[m")

if [ -d src ]
then 
 cd src/
else
 echo "$red Cannot find source ( src ) directory $normal" 
 exit 1
fi

echo " "
echo "Starting build process"
echo "---------------------------------"


# Make the automatic string scanners 
if [ -d StringRecognizer ]
then 
cd StringRecognizer
./make $@
./generateAmmServerScanners.sh
cd ..
fi


# Make each of the parts of the project!
# First of all the main library
if [ -d AmmServerlib ]
then 
cd AmmServerlib
./make $@
cd ..
fi
 
# Second the NULL version of the main library
if [ -d AmmServerNULLlib ]
then 
cd AmmServerNULLlib
./make $@
cd ..
fi
 
# Third the Captcha library that might be used by other projects 
if [ -d AmmCaptcha ]
then 
cd AmmCaptcha
./make $@
cd ..
fi

# Mini Clients go afterwards ----------------
if [ -d Services/SimpleTemplate ]
then 
cd Services/SimpleTemplate
./make $@
cd ..
cd ..
fi



if [ -d Services/AmmarServer ]
then 
cd Services/AmmarServer
./make $@
cd ..
cd ..
fi 

if [ -d Services/MyURL ]
then 
cd Services/MyURL
./make $@
cd ..
cd ..
fi 

if [ -d Services/MyLoader ]
then 
cd Services/MyLoader
./make $@
cd ..
cd ..
fi

if [ -d Services/ScriptRunner ]
then 
cd Services/ScriptRunner
./make $@
cd ..
cd ..
fi

if [ -d Services/GeoPosShare ]
then 
cd Services/GeoPosShare
./make $@
cd ..
cd ..
fi



# Unit Tests go in the end so that everything else is already there
if [ -d UnitTests ]
then 
cd UnitTests
./make $@
cd ..
fi




exit 0
