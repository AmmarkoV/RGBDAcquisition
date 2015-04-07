#!/bin/bash


OBJNUM="16"

function getStatistics 
{ #use R to generate statistics
  R -q -e "x <- read.csv('$1', header = F); summary(x); sd(x[ , 1])" > $2
  cat $1 | wc -l >> $2 
}  
 


if [ "$#" -ne 3 ]; then
    echo "Illegal groundTruth fileA fileB"
    echo "groundTruth , fileA and fileB should look like itemID timestamp X Y Z<newline>"
    exit 2
fi

 
#Switch to this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

rm Tmp/sumA.dat
rm Tmp/sumB.dat

count=0
while [ $count -le $OBJNUM ]
do
    IDNAME="O$count"P0

    ./comparePositionalData.sh $1 $2 $IDNAME 
    cat Results/distance$IDNAME$2.dat >>Tmp/sumA.dat 
    ./comparePositionalData.sh $1 $3 $IDNAME 
    cat Results/distance$IDNAME$3.dat >>Tmp/sumB.dat 
    (( count++ ))
done


getStatistics Tmp/sumA.dat Results/global$2-stats.txt
GNUPLOT_CMD="set terminal png; set output \"global$2.png\"; set ylabel \"Distance(mm)\"; plot \"Tmp/sumA.dat\"  with lines  title \"Movement Analysis, $2\""
gnuplot -e "$GNUPLOT_CMD"


getStatistics Tmp/sumB.dat Results/global$3-stats.txt
GNUPLOT_CMD="set terminal png; set output \"global$3.png\"; set ylabel \"Distance(mm)\"; plot \"Tmp/sumB.dat\"  with lines  title \"Movement Analysis, $3\""
gnuplot -e "$GNUPLOT_CMD"


exit 0
