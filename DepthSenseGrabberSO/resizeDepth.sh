#for file in ./depth*.pnm; do convert $file -resize 640x480 -compress none temp/`basename $file`; done
#for file in temp/depth*.pnm; do convert $file output_resize/`basename $file`; done
for file in ./depth*.pnm; do pnmenlarge 2 $file > output_resize/`basename $file`; done
