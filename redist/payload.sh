#!/bin/bash

exit 0

xdotool mousemove --sync 1 1
xdotool mousemove --sync 100 100



SERVICE="gpicview"
RESULT=`ps -a | sed -n /${SERVICE}/p`

if [ "${RESULT:-null}" = null ]; then
    echo "not running"
    echo `pwd`
    timeout 3 gpicview ./seeyou.jpg&
else
    echo "running"
fi
 
exit 0
