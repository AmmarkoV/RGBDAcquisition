#!/bin/bash

# Define the name of the text file to create
output_file="jpg_images.txt"

# Get the current working directory
DIR=$(pwd)

# Find all .jpg files in the current directory, append the full path, and write their names to the text file
find "$DIR" -maxdepth 1 -type f -iname "*.jpg" -exec echo "{}" \; > "$output_file"

# Print a message indicating completion
echo "List of .jpg images with full system paths has been saved to $output_file"

