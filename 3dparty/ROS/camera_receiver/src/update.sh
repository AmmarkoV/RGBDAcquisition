#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "Will now manually try to update all files.." 
echo "Wish me luck..! :)"


AMMCLIENT_REPOSITORY="https://raw.githubusercontent.com/AmmarkoV/AmmarServer/master/src/AmmClient"
LIST_OF_AMMCLIENT_LISTS="AmmClient.c AmmClient.h network.c network.h protocol.c protocol.h tools.c tools.h"

for file in $LIST_OF_AMMCLIENT_LISTS
do
    echo " Updating $file .. "
    rm $file
    wget $AMMCLIENT_REPOSITORY/$file
done




exit 0
