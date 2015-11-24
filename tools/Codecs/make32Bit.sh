#!/bin/bash

gcc -Wall -m32 -O2   utilConverterMain.c asciiInput.c codecs.c jpgExifexternal.c jpgExiforient_embed.c jpgInput.c pngInput.c ppmInput.c  -ljpeg -lpng -s   -o ./codecconverter32 


gcc -Wall -O2   utilConverterMain.c asciiInput.c codecs.c jpgExifexternal.c jpgExiforient_embed.c jpgInput.c pngInput.c ppmInput.c  -ljpeg -lpng -s   -o ./codecconverter64

 
exit 0
