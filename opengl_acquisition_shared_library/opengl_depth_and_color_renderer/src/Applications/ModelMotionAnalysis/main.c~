/** @file main.c
 *  @brief  A minimal binary that renders scene files using OGLRendererSandbox s
 *          X86 compilation: gcc -o -L/usr/X11/lib   main main.c -lGL -lX11 -lpng -ljpeg
 *          X64 compilation: gcc -o -L/usr/X11/lib64 main main.c -lGL -lX11 -lpng -ljpeg
 *  @author Ammar Qammaz (AmmarkoV)
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "../TrajectoryParser/InputParser_C.h"

#define MAX_JOINTS 128

struct Point3D
{
 float x;
 float y;
 float z;
};

struct staticStr
{
 char value[128];
};



struct Pose3D
{
 float x;
 float y;
 float z;

 float qW;
 float qX;
 float qY;
 float qZ;
};



struct motionStats
{
  unsigned int numberOfJoints;

  int startSet;
  struct Pose3D start;
  struct Pose3D min;
  struct Pose3D max;
  struct Pose3D end;

  struct Point3D startJoint[MAX_JOINTS];
  unsigned int numberOfSamples[MAX_JOINTS];

  struct Point3D previous[MAX_JOINTS];
  struct Point3D current[MAX_JOINTS];


  struct Point3D minimum[MAX_JOINTS];
  struct Point3D maximum[MAX_JOINTS];
  struct Point3D variability[MAX_JOINTS];
  struct staticStr name[MAX_JOINTS];

};


float _mma_abs(float value)
{
 if (value<0) { return -1 * value;}
 return value;
}

int _mma_floatEq(float element , float value )
{
 const float machineFloatPercision= 0.001;
 if ( element == value ) { return 1; }

 if ( ( value-machineFloatPercision<element ) && (( element<value+machineFloatPercision )) ) { return 1; }
 return 0;
}


int printMotionStats(struct motionStats * st)
{
  fprintf(stdout,"#!/bin/bash\n\n");
  unsigned int i=0;



fprintf(stdout,"POSE_MINIMUMS=\"%0.2f %0.2f %0.2f\" \n",st->min.x,st->min.y,st->min.z);
fprintf(stdout,"POSE_MAXIMUMS=\"%0.2f %0.2f %0.2f\" \n\n",st->max.x,st->max.y,st->max.z);

  for (i=0; i<st->numberOfJoints; i++)
  {
   fprintf(stdout,"min_%s=\"%0.2f %0.2f %0.2f\"\n",st->name[i].value,
                                               st->minimum[i].x ,
                                               st->minimum[i].y ,
                                               st->minimum[i].z
                                               );

   fprintf(stdout,"max_%s=\"%0.2f %0.2f %0.2f\"\n",st->name[i].value,
                                               st->maximum[i].x ,
                                               st->maximum[i].y ,
                                               st->maximum[i].z
                                               );

    if (
        (
        (_mma_floatEq(st->maximum[i].x-st->minimum[i].x,0.0)) &&
        (_mma_floatEq(st->maximum[i].y-st->minimum[i].y,0.0)) &&
        (_mma_floatEq(st->maximum[i].z-st->minimum[i].z,0.0)) &&
        (_mma_floatEq(st->minimum[i].x,0.0)) &&
        (_mma_floatEq(st->minimum[i].y,0.0)) &&
        (_mma_floatEq(st->minimum[i].z,0.0))
        ) ||
        (st->numberOfSamples[i]==0)
       )
    {
        fprintf(stdout,"var_%s=\"0 0 0\"\n",st->name[i].value);
    } else
    {
        fprintf(stdout,"var_%s=\"%0.2f %0.2f %0.2f\"\n",st->name[i].value,
                2*st->variability[i].x/st->numberOfSamples[i],
                2*st->variability[i].y/st->numberOfSamples[i],
                2*st->variability[i].z/st->numberOfSamples[i]
                );
    }

   fprintf(stdout,"\n");
  }

//---------------------------------------------------------
fprintf(stdout,"JOINT_MINIMUMS=\"");
  for (i=0; i<st->numberOfJoints; i++)
  {
    fprintf(stdout,"$min_%s ",st->name[i].value);
  }
fprintf(stdout,"\"\n\n");
//---------------------------------------------------------
fprintf(stdout,"JOINT_MAXIMUMS=\"");
  for (i=0; i<st->numberOfJoints; i++)
  {
    fprintf(stdout,"$max_%s ",st->name[i].value);
  }
fprintf(stdout,"\"\n\n");
//---------------------------------------------------------
fprintf(stdout,"JOINT_VARIANCE=\"");
  for (i=0; i<st->numberOfJoints; i++)
  {
    fprintf(stdout,"$var_%s ",st->name[i].value);
  }
fprintf(stdout,"\"\n\n");
//---------------------------------------------------------


fprintf(stdout,"JOINTS_NUMBER=\"%u\"\n",st->numberOfJoints);
//---------------------------------------------------------
fprintf(stdout,"JOINTS_IDS=\"");
  for (i=0; i<st->numberOfJoints; i++)
  {
    fprintf(stdout,"%s ",st->name[i].value);
  }
fprintf(stdout,"\"\n\n");
//---------------------------------------------------------
fprintf(stdout,"JOINTS_ENABLED=\"--joints $JOINTS_NUMBER $JOINTS_IDS\"\n\n");



fprintf(stdout,"POSE_START=\"%0.2f %0.2f %0.2f\" \n" ,st->start.x,st->start.y,st->start.z);
fprintf(stdout,"ROT_START=\"%0.5f %0.5f %0.5f %0.5f\" \n" ,st->start.qW,st->start.qX,st->start.qY,st->start.qZ);
for (i=0; i<st->numberOfJoints; i++)
  {
    fprintf(stdout,"%s=\"%0.2f %0.2f %0.2f\"\n",st->name[i].value,
                                                st->startJoint[i].x,
                                                st->startJoint[i].y,
                                                st->startJoint[i].z
                                               );
  }



//---------------------------------------------------------
fprintf(stdout,"\nJOINT_POSITION=\"");
  for (i=0; i<st->numberOfJoints; i++)
  {
    fprintf(stdout,"$%s ",st->name[i].value);
  }
fprintf(stdout,"\"\n\n\n\n");

return 1;

}


int getJointMemoryID(struct motionStats * st , const char * name , unsigned int * where2work )
{
  if (st->numberOfJoints>=MAX_JOINTS) { return 0; }

  if (st->numberOfJoints==0)
  { *where2work=0; } else
  {
    unsigned int i=0;
    for (i=0; i<st->numberOfJoints; i++)
    {
      if (strcmp(name,st->name[i].value)==0)
      {
       //fprintf(stdout,"Found it @ %s \n",name);
       *where2work=i;
       return 1;
      }
    }
  }
 *where2work = st->numberOfJoints;
 strncpy(st->name[*where2work].value,name,128);
 ++st->numberOfJoints;
 return 1;
}


int updateJoint(struct motionStats * st , unsigned int i, float x , float y , float z)
{
 if (x<st->minimum[i].x) { st->minimum[i].x=x; }
 if (y<st->minimum[i].y) { st->minimum[i].y=y; }
 if (z<st->minimum[i].z) { st->minimum[i].z=z; }

 if (x>st->maximum[i].x) { st->maximum[i].x=x; }
 if (y>st->maximum[i].y) { st->maximum[i].y=y; }
 if (z>st->maximum[i].z) { st->maximum[i].z=z; }


 if (st->numberOfSamples[i]==0)
 {
  st->startJoint[i].x=x;
  st->startJoint[i].y=y;
  st->startJoint[i].z=z;
 }

 if ( st->numberOfSamples[i]==0)
 {
   st->current[i].x=x;
   st->current[i].y=y;
   st->current[i].z=z;
 } else
 {
   st->previous[i].x=st->current[i].x;
   st->previous[i].y=st->current[i].y;
   st->previous[i].z=st->current[i].z;

   st->current[i].x=x;
   st->current[i].y=y;
   st->current[i].z=z;


   float diffX = _mma_abs(st->current[i].x - st->previous[i].x);
   float diffY = _mma_abs(st->current[i].y - st->previous[i].y);
   float diffZ = _mma_abs(st->current[i].z - st->previous[i].z);


   st->variability[i].x += diffX;
   st->variability[i].y += diffY;
   st->variability[i].z += diffZ;
 }

 st->numberOfSamples[i]+=1;

 return 1;
}


int updatePosition(struct motionStats * st , float x , float y , float z , float qW ,float qX , float qY , float qZ)
{
 if (x<st->min.x) { st->min.x=x; }
 if (y<st->min.y) { st->min.y=y; }
 if (z<st->min.z) { st->min.z=z; }

 if (x>st->max.x) { st->max.x=x; }
 if (y>st->max.y) { st->max.y=y; }
 if (z>st->max.z) { st->max.z=z; }

     if (!st->startSet)
     {
       st->min.x=x;
       st->min.y=y;
       st->min.z=z;

       st->max.x=x;
       st->max.y=y;
       st->max.z=z;


       st->startSet=1;
       st->start.x=x;
       st->start.y=y;
       st->start.z=z;

       st->start.qW=qW;
       st->start.qX=qX;
       st->start.qY=qY;
       st->start.qZ=qZ;
     }

 st->end.x=x;
 st->end.y=y;
 st->end.z=z;

 st->end.qW=qW;
 st->end.qX=qX;
 st->end.qY=qY;
 st->end.qZ=qZ;

 return 1;
}


int processCommand(struct InputParserC * ipc , struct motionStats * st ,char * line , unsigned int words_count , unsigned int startAtFrame)
{

  if (InputParser_WordCompareAuto(ipc,0,"MOVE"))
   {
      int frameNum = InputParser_GetWordInt(ipc,2);
      if (frameNum<startAtFrame) { return 0; }

      float x = InputParser_GetWordFloat(ipc,3);
      float y = InputParser_GetWordFloat(ipc,4);
      float z = InputParser_GetWordFloat(ipc,5);

      float qW = InputParser_GetWordFloat(ipc,6);
      float qX = InputParser_GetWordFloat(ipc,7);
      float qY = InputParser_GetWordFloat(ipc,8);
      float qZ = InputParser_GetWordFloat(ipc,9);

      updatePosition(st,x,y,z,qW,qX,qY,qZ);
   }
    else
  if (InputParser_WordCompareAuto(ipc,0,"POSE"))
   {
    int frameNum = InputParser_GetWordInt(ipc,2);
    if (frameNum<startAtFrame) { return 0; }

    //fprintf(stdout,"Found Frame %u \n",InputParser_GetWordInt(ipc,1));
    char str[128];
    if (InputParser_GetWord(ipc,3,str,128)!=0)
    {//Flush next frame
      float x = InputParser_GetWordFloat(ipc,4);
      float y = InputParser_GetWordFloat(ipc,5);
      float z = InputParser_GetWordFloat(ipc,6);

      unsigned int where2work=0;
      if  ( getJointMemoryID(st,str,&where2work) )
      {
        updateJoint(st,where2work,x,y,z);
      }


      //fprintf(stdout," %0.2f %0.2f %0.2f \n",x,y,z);
      return 1;
    }
   }

  return 0;
}



int appendFileMotionStats(char * filename,struct motionStats *st,struct InputParserC * ipc,unsigned int startAtFrame)
{
 char line [512]={0};

 fprintf(stderr,"Opening file %s\n",filename);
 FILE * fp = fopen(filename,"r");
 if (fp == 0 ) { fprintf(stderr,"Cannot open stream %s \n",filename); return 0; }


   while (!feof(fp))
   {
   //We get a new line out of the file
   int readOpResult = (fgets(line,512,fp)!=0);
   if ( readOpResult != 0 )
    {
      //We tokenize it
      unsigned int words_count = InputParser_SeperateWords(ipc,line,0);
      if ( words_count > 0 )
         {
             processCommand(ipc,st,line,words_count,startAtFrame);
         } // End of line containing tokens
    } //End of getting a line while reading the file
  }


  fclose(fp);
  return 1;
}






int main(int argc, char **argv)
{
 if (argc<2)
 {
     fprintf(stderr,"./ModelMotionAnalysis file1 file2 etc..\n");
     fprintf(stderr,"_______________________________________\n");
     return 0;
 }

 unsigned int startAtFrame =15;

 struct motionStats st={0};

 struct InputParserC * ipc=0;
 ipc = InputParser_Create(512,5);
 if (ipc==0)  { fprintf(stderr,"Cannot allocate memory for new stream\n");  return 0; }

  for (int i=1; i<argc; i++)
  {
    // fprintf(stderr,"%u - %s\n",i,argv[i]);
    appendFileMotionStats(argv[i],&st,ipc,startAtFrame);
  }

  printMotionStats(&st);

  InputParser_Destroy(ipc);

 return 0;
}
