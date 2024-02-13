#!/bin/bash

OVERLAY="$1"
BASE="$2"


ffmpeg -i $BASE -i $OVERLAY -filter_complex "[0:v][1:v]blend=all_expr='A*(1-0.5)+B*0.5'[outv]" -map "[outv]" -map 0:a? -c:a copy $OVERLAY_over_$BASE.webm



exit 0
