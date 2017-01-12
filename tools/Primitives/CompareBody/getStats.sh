#!/bin/bash

function getStatistics 
{ #use R to generate statistics
  R -q -e "x <- read.csv('$1', header = F); summary(x); sd(x[ , 1])" > $2
  cat $1 | wc -l >> $2 
}  


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"  

groundFile="$1/$1Ground.scene"
secondRun="$1/$1.scene"

./CompareBody --single $groundFile | grep "JointLengthDifference" | cut -d ' ' -f4 > $1/lengthdis.dat
./CompareBody --single $groundFile | grep "TotalDistance" | cut -d ' ' -f3 > $1/totaldis.dat
./CompareBody --single $groundFile | grep "JointDistance" | cut -d ' ' -f4 > $1/jointdis.dat
./CompareBody --single $groundFile | grep "PerJointDistance" | cut -d ' ' -f2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17   > $1/perjointdis.dat
  
./CompareBody --ground $groundFile $secondRun | grep "PerJointDistanceGround" | cut -d ' ' -f2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17   > $1/groundperjointdis.dat

 gnuplot -e "set terminal png size 800,600 enhanced font 'Verdana,14' linewidth 1; set output \"$1/$1lengthdis.png\"; set xlabel \"Frame Number\"; set ylabel \"Error in mm\"; plot \"$1/lengthdis.dat\" using 1 with lines title \"Distance for all joints \" "

 
 gnuplot -e "set terminal png size 800,600 enhanced font 'Verdana,14' linewidth 1; set output \"$1/$1totaldis.png\"; set xlabel \"Frame Number\"; set ylabel \"Total Error in mm\"; plot \"$1/totaldis.dat\" using 1 with lines title \"Total Error for all joints \" "

 gnuplot -e "set terminal png size 800,600 enhanced font 'Verdana,14' linewidth 1; set output \"$1/$1jointdis.png\"; set xlabel \"Frame Number\"; set ylabel \"Error in mm\"; plot \"$1/jointdis.dat\" using 1 with lines title \"Total Error for all joints \" "

 gnuplot -e "set terminal png size 800,600 enhanced font 'Verdana,14' linewidth 1;\
set output \"$1/$1perjointdis.png\";\
set xlabel \"Frame Number\";\
set ylabel \" Error in mm\"; \
set border linewidth 1;\
 plot \"$1/perjointdis.dat\" using 1 with lines title \"head\" ,\
  \"$1/perjointdis.dat\" using 2 with lines title \"neck\" ,\
  \"$1/perjointdis.dat\" using 3 with lines  title \"torso\",\
  \"$1/perjointdis.dat\" using 4 with lines  title \"right shoulder\",\
  \"$1/perjointdis.dat\" using 5 with lines title \"left shoulder\",\
  \"$1/perjointdis.dat\" using 6 with lines  title \"right elbow\",\
  \"$1/perjointdis.dat\" using 7 with lines  title \"left elbow\",\
  \"$1/perjointdis.dat\" using 8 with lines  title \"right hand\",\
  \"$1/perjointdis.dat\" using 9 with lines  title \"left hand\",\
  \"$1/perjointdis.dat\" using 10 with lines  title \"right hip\",\
  \"$1/perjointdis.dat\" using 11 with lines  title \"left hip\",\
  \"$1/perjointdis.dat\" using 12 with lines   title \"right knee\",\
  \"$1/perjointdis.dat\" using 13 with lines  title \"left knee\",\
  \"$1/perjointdis.dat\" using 14 with lines  title \"right foot\",\
  \"$1/perjointdis.dat\" using 15 with lines  title \"left foot\",\
  \"$1/perjointdis.dat\" using 16 with lines  title \"hip\"; "
 


 gnuplot -e "set terminal png size 800,600 enhanced font 'Verdana,14' linewidth 1;\
set output \"$1/$1groundperjointdis.png\";\
set xlabel \"Frame Number\";\
set ylabel \" Error in mm\"; \
set border linewidth 1;\
 plot \"$1/groundperjointdis.dat\" using 1 with lines title \"head\" ,\
  \"$1/groundperjointdis.dat\" using 2 with lines title \"neck\" ,\
  \"$1/groundperjointdis.dat\" using 3 with lines  title \"torso\",\
  \"$1/groundperjointdis.dat\" using 4 with lines  title \"right shoulder\",\
  \"$1/groundperjointdis.dat\" using 5 with lines title \"left shoulder\",\
  \"$1/groundperjointdis.dat\" using 6 with lines  title \"right elbow\",\
  \"$1/groundperjointdis.dat\" using 7 with lines  title \"left elbow\",\
  \"$1/groundperjointdis.dat\" using 8 with lines  title \"right hand\",\
  \"$1/groundperjointdis.dat\" using 9 with lines  title \"left hand\",\
  \"$1/groundperjointdis.dat\" using 10 with lines  title \"right hip\",\
  \"$1/groundperjointdis.dat\" using 11 with lines  title \"left hip\",\
  \"$1/groundperjointdis.dat\" using 12 with lines   title \"right knee\",\
  \"$1/groundperjointdis.dat\" using 13 with lines  title \"left knee\",\
  \"$1/groundperjointdis.dat\" using 14 with lines  title \"right foot\",\
  \"$1/groundperjointdis.dat\" using 15 with lines  title \"left foot\",\
  \"$1/groundperjointdis.dat\" using 16 with lines  title \"hip\"; "
 


exit 0
