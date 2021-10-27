#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

chmod 777 uploads/


cd uploads/
UPLOAD_DIR=`pwd`


cd /mnt
sudo mkdir stream
sudo chmod 777 stream

rm -rf uploads
cd $DIR 
ln -s /mnt/stream/ uploads
sudo chmod 777 uploads


echo "Please add to your /etc/fstab"
sudo mount -t tmpfs -o size=64m tmpfs /mnt/stream

echo "Please add to your /etc/fstab"
echo "/mnt/stream tmpfs   nodev,nosuid,noexec,nodiratime,size=64M   0 0"

$UPLOAD_DIR

exit 0
