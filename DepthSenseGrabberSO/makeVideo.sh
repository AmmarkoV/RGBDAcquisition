#!/bin/bash 

if (( $#==0 ))
then 
  echo "First argument: dataset (e.g. colorRaw), second: width times height (optional), third: extension (optional)"
  echo "Example: colorRaw pnm 640x480"
  exit 1
else
  if (( $#==1))
  then
    DATASET=$1
    FORMAT="640x480"
    EXTENSION="pnm"
  fi
  if (( $#==2))
  then
    DATASET=$1
    FORMAT=$2
    EXTENSION="pnm"
  fi
  if (( $#==3))
  then
    DATASET=$1
    FORMAT=$2
    EXTENSION=$3
  fi
fi

avconv -i $DATASET"Frame_0_"%05d.pnm -y -r 30 -threads 8 -b 30000k -s $FORMAT  video$DATASET".mp4" 
