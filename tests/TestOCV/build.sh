#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

g++ main.cpp -o TestOCV `pkg-config --libs --cflags opencv`

exit 0
