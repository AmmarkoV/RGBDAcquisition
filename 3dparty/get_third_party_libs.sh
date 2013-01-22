#!/bin/bash

if [ -d libfreenect ]
then
echo "Freenect appears to already exist .."
else
  echo "Do you want to download freenect ? " 
  echo
  echo -n " (Y/N)?"
  read answer
  if test "$answer" != "Y" -a "$answer" != "y";
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
         ln -s libfreenect/build/lib/libfreenect.so ../grabber/libfreenect.so
       fi
     if [ -e libfreenect/build/lib/libfreenect_sync.so ]
       then
         ln -s libfreenect/build/lib/libfreenect_sync.so ../grabber/libfreenect_sync.so
       fi

  fi
fi

if [ -d OpenNI1 ]
then
echo "OpenNI1 appears to already exist .."
else
  echo "Do you want to download freenect ? " 
  echo
  echo -n " (Y/N)?"
  read answer
  if test "$answer" != "Y" -a "$answer" != "y";
  then
     git clone git://github.com/OpenNI/OpenNI.git
     cd OpenNI
     mkdir build
     cd build
     cmake ..
     make
     cd ../../  

  fi
fi


if [ -d OpenNI2 ]
then
echo "OpenNI2 appears to already exist .."
else
  echo "Do you want to download freenect ? " 
  echo
  echo -n " (Y/N)?"
  read answer
  if test "$answer" != "Y" -a "$answer" != "y";
  then
     git clone git://github.com/OpenNI/OpenNI2.git
     cd OpenNI2
     mkdir build
     cd build
     cmake ..
     make
     cd ../../ 
     ln -s OpenNI2/Redist/OpenNI2 ../grabber/OpenNI2 
     ln -s OpenNI2/Redist/OpenNI.ini ../grabber/OpenNI.ini 
     ln -s OpenNI2/Redist/PS1080.ini ../grabber/PS1080.ini
     ln -s OpenNI2/Redist/libOpenNI2.so ../grabber/libOpenNI2.so

  fi
fi

exit 0
