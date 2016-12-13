#!/bin/bash

function getStatistics 
{ #use R to generate statistics
  R -q -e "x <- read.csv('$1', header = F); summary(x); sd(x[ , 1])" > $2
  cat $1 | wc -l >> $2 
}  


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"  

./CompareBody lastrun.scene | grep "TotalDistance" | cut -d ' ' -f3 > totaldis.dat
./CompareBody lastrun.scene | grep "JointDistance" | cut -d ' ' -f4 > jointdis.dat
./CompareBody lastrun.scene | grep "PerJointDistance" | cut -d ' ' -f2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17   > perjointdis.dat
  

 gnuplot -e "set terminal png size 800,600 enhanced font 'Verdana,14' linewidth 1; set output \"totaldis.png\"; set xlabel \"Frame Number\"; set ylabel \"Total Error in mm\"; plot \"totaldis.dat\" using 1 with lines title \"Total Error for all joints \" "

 gnuplot -e "set terminal png size 800,600 enhanced font 'Verdana,14' linewidth 1; set output \"jointdis.png\"; set xlabel \"Frame Number\"; set ylabel \"Error in mm\"; plot \"jointdis.dat\" using 1 with lines title \"Total Error for all joints \" "

 gnuplot -e "set terminal png size 800,600 enhanced font 'Verdana,14' linewidth 1;\
set output \"perjointdis.png\";\
set xlabel \"Frame Number\";\
set ylabel \" Error in mm\"; \
set border linewidth 1;\
 plot \"perjointdis.dat\" using 1 with lines title \"head\" ,\
  \"perjointdis.dat\" using 2 with lines title \"neck\" ,\
  \"perjointdis.dat\" using 3 with lines  title \"torso\",\
  \"perjointdis.dat\" using 4 with lines  title \"right shoulder\",\
  \"perjointdis.dat\" using 5 with lines title \"left shoulder\",\
  \"perjointdis.dat\" using 6 with lines  title \"right elbow\",\
  \"perjointdis.dat\" using 7 with lines  title \"left elbow\",\
  \"perjointdis.dat\" using 8 with lines  title \"right hand\",\
  \"perjointdis.dat\" using 9 with lines  title \"left hand\",\
  \"perjointdis.dat\" using 10 with lines  title \"right hip\",\
  \"perjointdis.dat\" using 11 with lines  title \"left hip\",\
  \"perjointdis.dat\" using 12 with lines   title \"right knee\",\
  \"perjointdis.dat\" using 13 with lines  title \"left knee\",\
  \"perjointdis.dat\" using 14 with lines  title \"right foot\",\
  \"perjointdis.dat\" using 15 with lines  title \"left foot\",\
  \"perjointdis.dat\" using 16 with lines  title \"hip\"; "
 
exit 0
