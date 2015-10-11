#!/bin/bash
sudo echo "Starting Bluetooth"
hciconfig
hcitool scan
sudo killall gpsd
sudo service gpsd stop
sudo rfcomm connect 0 00:19:EF:11:16:ED
#sudo usermod -a -G dialout ammar
sudo chmod 777 /dev/rfcomm0

#sudo rfcomm release 0
exit 0
