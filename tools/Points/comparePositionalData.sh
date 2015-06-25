#!/bin/bash 
function getStatistics 
{ #use R to generate statistics ( sudo apt-get install r-base )
  R -q -e "x <- read.csv('$1', header = F); summary(x); sd(x[ , 1])" > $2
  cat $1 | wc -l >> $2 
}  
 
if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters , should be 3 fileA fileB itemID"
    echo "fileA and fileB should look like itemID timestamp X Y Z<newline>"
    exit 2
fi
 
#Switch to this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

mkdir Results &> /dev/null
mkdir Tmp  &> /dev/null

OUTPUT_IMAGE="Results/$3$2.png" 
MOVEMENT_FILE="Results/distance$3$2.dat" 

cat $1 | grep $3 | cut -d ' ' -f3,4,5 > Tmp/fileAPos.dat
cat $2 | grep $3 | cut -d ' ' -f3,4,5 > Tmp/fileBPos.dat
rm $MOVEMENT_FILE  &> /dev/null

echo "Comparing ID $3 of $1 to $2" 

while read -u 3 -r lineA && read -u 4 -r lineB; do
 ./PointsDistance $lineA $lineB >> $MOVEMENT_FILE 
done 3<Tmp/fileAPos.dat 4<Tmp/fileBPos.dat

GNUPLOT_CMD="set terminal png; set output \"$OUTPUT_IMAGE\"; set ylabel \"Distance(mm)\"; plot \"$MOVEMENT_FILE\"  with lines  title \"Movement Analysis, $1\""
gnuplot -e "$GNUPLOT_CMD"

getStatistics $MOVEMENT_FILE Results/$3$2-stats.txt

#echo "done."

exit 0
