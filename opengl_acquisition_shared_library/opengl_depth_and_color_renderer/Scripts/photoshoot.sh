#!/bin/bash
THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$THISDIR"
cd ..

./Renderer --from Scenes/photo.conf --photo 1 0 0 0 16 16 --size 1024 1024
timeout 5 gpicview color.pnm
./imageopsutility color.pnm --learn 16 16
Scripts/make_video.sh
