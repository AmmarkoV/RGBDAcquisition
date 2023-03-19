#!/bin/bash

BVHFILE="$1"
BVHMOTIONID="$2"

grep -A10000 -P '^Frame Time: 0.04166667$' $BVHFILE | cut -d ' ' -f $BVHMOTIONID 


exit 0
