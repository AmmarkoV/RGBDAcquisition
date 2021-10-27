#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

chmod 777 uploads/


cd uploads/
UPLOAD_DIR=`pwd`


echo "Please add to your /etc/fstab"
sudo mount -t tmpfs -o size=64m tmpfs $UPLOAD_DIR

echo "Please add to your /etc/fstab"
echo "$UPLOAD_DIR tmpfs   nodev,nosuid,noexec,nodiratime,size=64M   0 0"

exit 0
