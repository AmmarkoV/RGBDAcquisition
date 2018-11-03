#!/bin/bash


CURDIR=`pwd`


SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPTDIR"

cd ../../../tools/DepthImagesConverter/
cd ../../tools/DepthImagesConverter/
DEPTHTOOLDIR=`pwd`

if [ -e $DEPTHTOOLDIR/DepthImagesConverter ]
then
 echo "Found Depth Image Converter.. Good to go.!"
else
 echo "Could not find depth image converter please compile RGBDAcquisition..!" 
 exit 0
fi

cd $CURDIR

echo "$@"
for inputDataset in $@
do

echo "Doing $inputDataset" 
cd $CURDIR

MAXRUNJOBS=12

INITIALSIZE=`du -hs $inputDataset | cut -f1`

CONVNAME="tmp_under_conversion-$BASHPID"
cd $inputDataset



COLORFILENUM=`ls -al | grep color | grep .pnm |  wc -l`
DEPTHFILENUM=`ls -al | grep depth | grep .pnm |  wc -l`
DOCONV=1

if [ $COLORFILENUM -eq 0 ]; 
then
 echo "No color pnm files found , $inputDataset is not a valid PNM dataset for compression..!"
 DOCONV=0  
elif [ $DEPTHFILENUM -eq 0 ]; 
then
 echo "No depth pnm files found , $inputDataset is not a valid PNM dataset for compression..!"
 DOCONV=0
fi


if [ $DOCONV -eq 1 ]; 
then

echo "Looks $inputDataset is an uncompressed dataset ($COLORFILENUM pnm color files , $DEPTHFILENUM pnm depth files ) will try to compress it now "


mkdir $SCRIPTDIR/$CONVNAME

cp *.calib $SCRIPTDIR/$CONVNAME/
cp ./*.sh $SCRIPTDIR/$CONVNAME/
cp ./*.mp4 $SCRIPTDIR/$CONVNAME/

echo "Converting Color Files"

FILES_TO_CONVERT=`ls | grep color | grep .pnm`


for f in $FILES_TO_CONVERT
do 
 RUNNINGJOBS=`ps -A | grep convert | wc -l`
 while [ $RUNNINGJOBS -gt $MAXRUNJOBS ]
  do
   sleep 0.03
   echo -n "@" 
   RUNNINGJOBS=`ps -A | grep convert | wc -l`
  done 

 TARGETNAME=`basename $f .pnm`
 convert $f $SCRIPTDIR/$CONVNAME/$TARGETNAME.jpg&
 echo -n "."
done


echo "Converting Depth Files"

FILES_TO_CONVERT=`ls | grep depth | grep .pnm`
for f in $FILES_TO_CONVERT
do 

 RUNNINGJOBS=`ps -A | grep DepthImagesCon | wc -l`
 while [ $RUNNINGJOBS -gt $MAXRUNJOBS ]
  do
   sleep 0.03
   echo -n "@" 
   RUNNINGJOBS=`ps -A | grep DepthImagesCon | wc -l`
  done 


 TARGETNAME=`basename $f .pnm`
 $DEPTHTOOLDIR/DepthImagesConverter "$f" "$SCRIPTDIR/$CONVNAME/$TARGETNAME.png"&
done

cd ..

sleep 1
clear

FINALSIZE=`du -hs $SCRIPTDIR/$CONVNAME| cut -f1`
  echo
  echo   
echo "Compressed $inputDataset from $INITIALSIZE to $FINALSIZE"


echo "Overwrite old dataset ?"
  echo
  echo   
  echo ""
  echo
  echo -n "            (Y/N)? : "
  read answer
  if test "$answer" != "Y" -a "$answer" != "y";
  then 
   echo "Naming the compressed dataset $CURDIR/$inputDataset-Compressed.." 
   mv $SCRIPTDIR/$CONVNAME "$CURDIR/$inputDataset-Compressed" 
else
   echo "Overwriting.." 
   rm -rf $CURDIR/$inputDataset
   mv $SCRIPTDIR/$CONVNAME $CURDIR/$inputDataset 
fi

fi
 
done



exit 0
