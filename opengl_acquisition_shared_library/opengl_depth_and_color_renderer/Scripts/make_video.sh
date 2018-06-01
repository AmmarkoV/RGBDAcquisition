#!/bin/bash 

  avconv -i tile-%05d.jpg -r 15 -threads 8 -b 30000k -s 1631x1080  outHD.mp4 
  avconv -i outHD.mp4  -r 6 -pix_fmt rgb24 "outHD-temp.gif" 
  convert -layers Optimize "outHD-temp.gif" "outHD.gif"

exit 0

