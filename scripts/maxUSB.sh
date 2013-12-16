#!/bin/bash

sudo modprobe usbcore usbfs_memory_mb=1000
sudo nice -n -20 ionice -c 1 -n 0 $@

exit 0
