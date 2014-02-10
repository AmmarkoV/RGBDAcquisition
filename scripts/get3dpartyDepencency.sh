#!/bin/bash

#Ok this small script does automatic library detection and 
#decides wether to link ( or not ) to our 3d party dependencies
#that way our make script , if we can we try to link to system wide libraries
#like -openni , then go local libraries , and if that fails our build log gets appended
#with the failure informatiion and the lib gets compiled without a fuss :)
#and everyone is happy 

EXPECTED_ARGS=1
E_BADARGS=65
FLAGTOENABLE="-L." 
LIBSTOLINK=""

if [ $# -ne $EXPECTED_ARGS ]
then
  echo "Usage: `basename $0` {arg}"
  exit $E_BADARGS
fi


if [ "$1" == "freenect" ]
then 
  if [ -e "libfreenect.so" ]; then LIBSTOLINK="$LIBSTOLINK ./libfreenect.so"; fi
  if [ -e "libfreenect_sync.so" ]; then LIBSTOLINK="$LIBSTOLINK ./libfreenect_sync.so"; fi
  if [ -e "/usr/local/lib64/libfreenect.so" ]; then LIBSTOLINK="$LIBSTOLINK /usr/local/lib64/libfreenect.so"; fi
  if [ -e "/usr/local/lib64/libfreenect_sync.so" ]; then LIBSTOLINK="$LIBSTOLINK /usr/local/lib64/libfreenect_sync.so"; fi
  #if [ -e "libfreenect.so.0.1" ]; then LIBSTOLINK="$LIBSTOLINK ./libfreenect.so.0.1"; fi
  #if [ -e "libfreenect_sync.so.0.1" ]; then LIBSTOLINK="$LIBSTOLINK ./libfreenect_sync.so.0.1"; fi

  if [ -z "$LIBSTOLINK" ]; then echo "No libraries found"; else
                                echo "$FLAGTOENABLE $LIBSTOLINK"
  fi
 
elif [ "$1" == "openni1" ]
then 
 if [ -e "libOpenNI.so" ]; then LIBSTOLINK="$LIBSTOLINK ./libOpenNI.so"; fi
 if [ -e "libOpenNI.jni.so" ]; then LIBSTOLINK="$LIBSTOLINK ./libOpenNI.jni.so"; fi 
 if [ -e "libnimRecorder.so" ]; then LIBSTOLINK="$LIBSTOLINK ./libnimRecorder.so"; fi 
 if [ -e "libnimMockNodes.so" ]; then LIBSTOLINK="$LIBSTOLINK ./libnimMockNodes.so"; fi 
 if [ -e "libnimCodecs.so" ]; then LIBSTOLINK="$LIBSTOLINK ./libnimCodecs.so"; fi  
 
  if [ -z "$LIBSTOLINK" ]; then echo "No libraries found"; else
                                echo "$FLAGTOENABLE $LIBSTOLINK"
  fi

elif [ "$1" == "openni2" ]
then 
  if [ -e "libOpenNI2.so" ] 
  then 
         LIBSTOLINK="$LIBSTOLINK ./libOpenNI2.so" 
         echo "$FLAGTOENABLE $LIBSTOLINK"
  fi
fi 

exit 0
