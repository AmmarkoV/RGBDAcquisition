#!/bin/bash
  
 
./run_viewer.sh -module OPENNI2 -from 0 -processor ../processors/BodyTracker/libBodyTracker.so  BodyTracker $@
  
exit 0
