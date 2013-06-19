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
if [ -d MyURL ]
then 
cd SimpleTemplate
./make $@
cd ..
fi

if [ -d MyURL ]
then 
cd MyURL
./make $@
cd ..
fi 

if [ -d MyLoader ]
then 
cd MyLoader
./make $@
cd ..
fi

if [ -d ScriptRunner ]
then 
cd ScriptRunner
./make $@
cd ..
fi

# Unit Tests go in the end so that everything else is already there
if [ -d UnitTests ]
then 
cd UnitTests
./make $@
cd ..
fi

#AmmarServer main executable in the end
./make $@
cd ..


exit 0
