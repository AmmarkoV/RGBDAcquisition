#!/bin/bash 
#JtHipRt JtKneeRt JtAnkleRt JtHipLf JtKneeLf JtAnkleLf JtSpineA JtSpineB JtNeckB JtUpperFaceParent JtShoulderRt JtElbowRt JtWristRt JtShoulderLf JtElbowLf JtWristLf
#JtLowerNoseLf  JtNeckB  JtSpineB  JtShoulderRt  JtShoulderLf  JtElbowRt  JtElbowLf  JtWristRt  JtWristLf  JtHipRt  JtHipLf  JtKneeRt  JtKneeLf  JtAnkleRt  JtAnkleLf  JtSpineA
function getarg {
                  echo "$1" |tr '\r' ' ' |  cut -d ' ' -f$2   
                }  

i=0
joints=""
while IFS='' read -r line || [[ -n "$line" ]]; do
#echo "Line $i: $line"
if [ $i -ne 0 ]; then
 X=`echo $line | cut -d ' ' -f1`
 Y=`echo $line | cut -d ' ' -f2`
 Z=`echo $line | cut -d ' ' -f3`
 qW=`echo $line | cut -d ' ' -f4`
 qX=`echo $line | cut -d ' ' -f5`
 qY=`echo $line | cut -d ' ' -f6`
 qZ=`echo $line | cut -d ' ' -f7`
 echo "MOVE(human,$i,$X,$Y,$Z,$qW,$qX,$qY,$qZ)"  
 #echo "MOVE(human,$i,-19.231,-54.976,2299.735,0.707107,0.707107,0.000000,0.0)"  

 
 jNum=8 
 for j in {1..16..1}
  do 
  sJoint=`getarg "$joints" $j`
  X=`getarg "$line" $jNum` 
  ((jNum=jNum+1))
  Y=`getarg "$line" $jNum` 
  ((jNum=jNum+1))
  Z=`getarg "$line" $jNum` 
  ((jNum=jNum+1))
   
  echo "POSE(human,$i,$sJoint,$X,$Y,$Z)"
 done
 echo ""
else
 #Write header once in the start 
 echo "AUTOREFRESH(1500)"
 echo "BACKGROUND(0,0,0)"
 echo "SCALE_WORLD(-0.01,-0.01,0.01)"
 echo "MAP_ROTATIONS(-1,-1,1,zxy)"
 echo "OFFSET_ROTATIONS(0,0,0)"
 echo "EMULATE_PROJECTION_MATRIX(535.423889 , 0.0 , 320.0 , 0.0 , 533.48468, 240.0 , 0 , 1)"

 echo "SILENT(1)"
 echo "RATE(100)"
 echo "INTERPOLATE_TIME(1)"
 echo "MOVE_VIEW(1)"

 echo "OBJECT_TYPE(humanMesh,Models/AmmarH.tri)"
 echo "RIGID_OBJECT(human,humanMesh, 255,0,0,0,0 ,10.0,10.0,10.0)" 
 echo "INTERPOLATE_TIME(1)" 
 #And get the joint list 
 joints="$line "
fi

((i=i+1))
done < "$1"



exit 0
