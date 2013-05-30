#!/bin/bash

BINARIES_THAT_NEED_LIBS="grabber grabber_mux grabber_segment acquisitionBroadcast"

if [ -d libfreenect ]
then
echo "Freenect appears to already exist .."
else
  echo "Do you want to download freenect ? " 
  echo
  echo -n " (Y/N)?"
  read answer
  if test "$answer" != "N" -a "$answer" != "n";
  then
     git clone git://github.com/OpenKinect/libfreenect.git
     cd libfreenect
     mkdir build
     cd build
     cmake ..
     make
     cd ../../ 

     if [ -e libfreenect/build/lib/libfreenect.so ]
       then

         for f in BINARIES_THAT_NEED_LIBS
           do  
             ln -s libfreenect/build/lib/libfreenect.so ../$f/libfreenect.so
           done
 
     fi
     if [ -e libfreenect/build/lib/libfreenect_sync.so ]
       then

         for f in BINARIES_THAT_NEED_LIBS
           do  
             ln -s libfreenect/build/lib/libfreenect_sync.so ../$f/libfreenect_sync.so
           done
     fi

  fi
fi

if [ -d OpenNI ]
then
echo "OpenNI1 appears to already exist .."
else
  echo "Do you want to download freenect ? " 
  echo
  echo -n " (Y/N)?"
  read answer
  if test "$answer" != "N" -a "$answer" != "n";
  then
     git clone git://github.com/OpenNI/OpenNI.git 
     cd OpenNI/Platform/Linux/CreateRedist
     ./RedistMaker
     
     cd ../Redist
     #sudo ./install.sh 
 
     cd ../../../../
  fi
fi


if [ -d OpenNI2 ]
then
echo "OpenNI2 appears to already exist .."
else
  echo "Do you want to download it ? " 
  echo
  echo -n " (Y/N)?"
  read answer
  if test "$answer" != "N" -a "$answer" != "n";
  then
     git clone git://github.com/OpenNI/OpenNI2.git
     cd OpenNI2
     make 
     cd ..
     
     #should be at 3dparty dir  
     for f in BINARIES_THAT_NEED_LIBS
           do  
               ln -s OpenNI2/Bin/x64-Release/OpenNI2 ../$f/OpenNI2 
               ln -s OpenNI2/Config/OpenNI.ini ../$f/OpenNI.ini 
               ln -s OpenNI2/Config/PS1080.ini ../$f/PS1080.ini
               ln -s OpenNI2/Bin/x64-Release/libOpenNI2.so ../$f/libOpenNI2.so
           done


  fi
fi

exit 0
