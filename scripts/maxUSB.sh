#!/bin/bash

#Set maximum USB Performance , and allocate a LOT of USB space
sudo modprobe usbcore usbfs_memory_mb=1000

#Run what we wanted to run at top priority 
sudo nice -n -20 ionice -c 1 -n 0 $@

exit $?
