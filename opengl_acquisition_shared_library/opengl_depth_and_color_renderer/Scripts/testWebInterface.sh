#!/bin/bash

wget -qO- "http://127.0.0.1:8080/control.html?x=$1&y=$2&z=$3&qX=0&qY=0&qZ=0" &> /dev/null




exit 0
