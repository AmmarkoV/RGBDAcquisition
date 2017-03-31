#!/bin/bash
valgrind --tool=callgrind  ./Renderer $@
exit 0
