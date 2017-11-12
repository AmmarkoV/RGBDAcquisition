#!/bin/bash
THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$THISDIR"
cd ..

./Renderer --from Scenes/photo.conf --photo 1 0 0 0 32 32 --size 4096 4096
timeout 5 gpicview color.pnm
exit 0
