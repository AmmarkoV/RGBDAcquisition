#!/bin/bash

#./BVHTester --from $1 --offsetPositionRotation 0 750 2000 0 0 0 --occlusions  --svg svg # --bvh $outputDir/$f-random.bvh


./BVHTester --from $1 --offsetPositionRotation 0 750 2000 0 0 0 --mirror rCollar lCollar --occlusions  --svg svg # --bvh $outputDir/$f-random.bvh


exit 0
