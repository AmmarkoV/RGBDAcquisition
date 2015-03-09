#for file in ./color*.pnm; do convert $file -resize 640x480 -compress none temp/`basename $file`; done
#for file in temp/color*.pnm; do convert $file output_resize/`basename $file`; done
for file in ./color*.pnm; do pnmenlarge 2 $file > output_resize/`basename $file`; done
