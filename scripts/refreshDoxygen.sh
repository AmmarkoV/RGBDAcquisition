#!/bin/bash
doxygen doc/doxyfile
cd doc/latex
make
cd .. 
cd ..

ln -s doc/latex/refman.pdf RGBDAcquisition.pdf

exit 0
