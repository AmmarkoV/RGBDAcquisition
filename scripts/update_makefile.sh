#!/bin/bash

#uses cbp2make , get it here - > http://sourceforge.net/projects/cbp2make/
cbp2make -in CodeBlocks.workspace -unix -out makefile

cd tools
cbp2make -in Tools.workspace -unix -out makefile
cd ..

exit 0
