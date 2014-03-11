#!/bin/bash

SOURCEDIR="$1"
TARGETDIR="$2"

mkdir $TARGETDIR

cp $SOURCEDIR/colorFrame_0_00000.pnm $TARGETDIR/colorFrame_0_00000.pnm
cp $SOURCEDIR/depthFrame_0_00000.pnm $TARGETDIR/depthFrame_0_00000.pnm

cp $SOURCEDIR/color.calib $TARGETDIR/color.calib
cp $SOURCEDIR/depth.calib $TARGETDIR/depth.calib


exit 0
