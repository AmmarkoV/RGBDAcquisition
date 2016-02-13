#!/bin/bash
sudo echo "Starting Bluetooth"
hciconfig
hcitool scan
sudo service gpsd stop
sudo killall gpsd

#sudo rfcomm connect 0 00:19:EF:11:16:ED
#sudo rfcomm connect 0 44:6D:6C:17:3A:13 #connect to mobile

sudo ./bypassBTBS.sh /dev/rfcomm0&
#sudo usermod -a -G dialout ammar
#sudo usermod -a -G dialout gpsd
#sudo chmod 777 /dev/rfcomm0


#sudo service gpsd start
sudo gpsd -b -N -D 5 /dev/myGPSA&


delay 2
xgps
#sudo rfcomm release 0
#sudo rfcomm release hci0 

exit 0
