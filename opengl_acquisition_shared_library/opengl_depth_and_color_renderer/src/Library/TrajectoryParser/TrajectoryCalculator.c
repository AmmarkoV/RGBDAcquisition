#include "TrajectoryCalculator.h"
#include "TrajectoryParserDataStructures.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define PI 3.141592653589793238462643383279502884197


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */


#include "../../../../../tools/AmMatrix/matrix4x4Tools.h"
#include "../../../../../tools/AmMatrix/matrixCalculations.h"
#include "../../../../../tools/AmMatrix/quaternions.h"



int accessOfObjectPositionIsOk(struct VirtualStream * stream,unsigned int ObjID,unsigned int FrameIDToReturn)
{
   if (stream==0)                      {  fprintf(stderr,"movePositionOfObjectTrajectory : Cannot access stream\n"); return 0; }
   if (stream->object==0)              {  fprintf(stderr,"movePositionOfObjectTrajectory : Cannot access objects\n"); return 0; }
   if (stream->numberOfObjects<=ObjID) {  fprintf(stderr,"movePositionOfObjectTrajectory : Cannot access objects index %u\n",ObjID); return 0; }

   if (stream->object[ObjID].frame==0) {  fprintf(stderr,"movePositionOfObjectTrajectory : Cannot Access frames for object %u \n",ObjID); return 0; }
   if (stream->object[ObjID].numberOfFrames==0 )
   {
     fprintf(stderr,"Position %u of Object %u cannot be altered since we only have %u positions \n",FrameIDToReturn,ObjID,stream->object[ObjID].numberOfFrames);
     return 0;
   }

   if (FrameIDToReturn>=stream->object[ObjID].numberOfFrames) { fprintf(stderr,"Position %u of Object %u is out of bounds\n" , FrameIDToReturn , ObjID); return 0; }
 return 1;
}


int movePositionOfObjectTrajectorySt(struct VirtualStream * stream,unsigned int ObjID,unsigned int FrameIDToReturn,float relX,float relY,float relZ)
{
    FrameIDToReturn=FrameIDToReturn%stream->object[ObjID].numberOfFrames;
    if (!accessOfObjectPositionIsOk(stream,ObjID,FrameIDToReturn)) { return 0; }

    stream->object[ObjID].frame[FrameIDToReturn].x+=relX;
    stream->object[ObjID].frame[FrameIDToReturn].y+=relY;
    stream->object[ObjID].frame[FrameIDToReturn].z+=relZ;
    fprintf(stderr,"This doesnt work for some weird reason.. Moving Object %u at %u/%u by %0.2f %0.2f %0.2f ( %0.2f %0.2f %0.2f  )\n",ObjID,FrameIDToReturn,stream->object[ObjID].numberOfFrames,relX,relY,relZ,
            stream->object[ObjID].frame[FrameIDToReturn].x,stream->object[ObjID].frame[FrameIDToReturn].y,stream->object[ObjID].frame[FrameIDToReturn].z);
 return 1;
}

int printObjectTrajectory(struct VirtualStream * stream,unsigned int ObjID,unsigned int FrameIDToReturn)
{
    FrameIDToReturn=FrameIDToReturn%stream->object[ObjID].numberOfFrames;
    if (!accessOfObjectPositionIsOk(stream,ObjID,FrameIDToReturn)) { return 0; }

    fprintf(stderr,"printObjectTrajectory quaternions are qXqYqZqW\n");
    float euler[3];
    float quaternions[4];

    fprintf(stderr,"POS=\"%0.2f %0.2f %0.2f\"\n",
            stream->object[ObjID].frame[FrameIDToReturn].x,
            stream->object[ObjID].frame[FrameIDToReturn].y,
            stream->object[ObjID].frame[FrameIDToReturn].z);

   if ( stream->object[ObjID].frame[FrameIDToReturn].isQuaternion )
   {
    quaternions[0]=stream->object[ObjID].frame[FrameIDToReturn].rot1;
    quaternions[1]=stream->object[ObjID].frame[FrameIDToReturn].rot2;
    quaternions[2]=stream->object[ObjID].frame[FrameIDToReturn].rot3;
    quaternions[3]=stream->object[ObjID].frame[FrameIDToReturn].rot4;
    quaternions2Euler(euler,quaternions,1);
   } else
   {
    euler[0]=stream->object[ObjID].frame[FrameIDToReturn].rot1;
    euler[1]=stream->object[ObjID].frame[FrameIDToReturn].rot2;
    euler[2]=stream->object[ObjID].frame[FrameIDToReturn].rot3;
    euler2Quaternions(quaternions,euler,1);
   }

  fprintf(stderr,"ROT=\"%0.2f %0.2f %0.2f\"\n", euler[0],euler[1], euler[2] );
  fprintf(stderr,"QUAT=\"%f %f %f %f\"\n", quaternions[0],quaternions[1], quaternions[2], quaternions[3] );


 return 1;
}

int movePositionOfObjectTrajectory(struct VirtualStream * stream,unsigned int ObjID,unsigned int FrameIDToReturn,float * relX,float * relY,float * relZ)
{
    FrameIDToReturn=FrameIDToReturn%stream->object[ObjID].numberOfFrames;
    if (!accessOfObjectPositionIsOk(stream,ObjID,FrameIDToReturn)) { return 0; }

    stream->object[ObjID].frame[FrameIDToReturn].x+=*relX;
    stream->object[ObjID].frame[FrameIDToReturn].y+=*relY;
    stream->object[ObjID].frame[FrameIDToReturn].z+=*relZ;
    fprintf(stderr,"Moving Object %u at %u/%u by %0.2f %0.2f %0.2f ( %0.2f %0.2f %0.2f  )\n",ObjID,FrameIDToReturn,stream->object[ObjID].numberOfFrames,*relX,*relY,*relZ,
            stream->object[ObjID].frame[FrameIDToReturn].x,stream->object[ObjID].frame[FrameIDToReturn].y,stream->object[ObjID].frame[FrameIDToReturn].z);
 return 1;
}


int rotatePositionOfObjectTrajectory(struct VirtualStream * stream,unsigned int ObjID,unsigned int FrameIDToReturn,float *x,float *y,float *z,float *angleDegrees)
{
   FrameIDToReturn=FrameIDToReturn%stream->object[ObjID].numberOfFrames;
   if (!accessOfObjectPositionIsOk(stream,ObjID,FrameIDToReturn)) { return 0; }

   if ( stream->object[ObjID].frame[FrameIDToReturn].isQuaternion )
   {
       fprintf(stderr,"rotatePositionOfObjectTrajectory doing a quaternion rotation %0.2f %0.2f %0.2f , angle %0.2f..\n",*x,*y,*z,*angleDegrees);
       float quaternion[4]={0};
       quaternion[0]=stream->object[ObjID].frame[FrameIDToReturn].rot1;
       quaternion[1]=stream->object[ObjID].frame[FrameIDToReturn].rot2;
       quaternion[2]=stream->object[ObjID].frame[FrameIDToReturn].rot3;
       quaternion[3]=stream->object[ObjID].frame[FrameIDToReturn].rot4;
       //-----------
        quaternionRotate(quaternion,*x,*y,*z,*angleDegrees,0);
       //-----------
       stream->object[ObjID].frame[FrameIDToReturn].rot1=quaternion[0];
       stream->object[ObjID].frame[FrameIDToReturn].rot2=quaternion[1];
       stream->object[ObjID].frame[FrameIDToReturn].rot3=quaternion[2];
       stream->object[ObjID].frame[FrameIDToReturn].rot4=quaternion[3];
   } else
   {
     fprintf(stderr,"rotatePositionOfObjectTrajectory doing an euler rotation %0.2f %0.2f %0.2f , angle %0.2f..\n",*x,*y,*z,*angleDegrees);
     if ( (*x==1.0) && (*y==0.0) && (*z==0.0) ) { stream->object[ObjID].frame[FrameIDToReturn].rot1+=*angleDegrees; } else
     if ( (*x==0.0) && (*y==1.0) && (*z==0.0) ) { stream->object[ObjID].frame[FrameIDToReturn].rot2+=*angleDegrees; } else
     if ( (*x==0.0) && (*y==0.0) && (*z==1.0) ) { stream->object[ObjID].frame[FrameIDToReturn].rot3+=*angleDegrees; }
   }


 return 0;
}


int smoothTrajectoriesOfObject(struct VirtualStream * stream,unsigned int ObjID)
{
  float avg=0.0;
  unsigned int pos=0;
  for (pos=1; pos<stream->object[ObjID].numberOfFrames; pos++)
  {
    //------------------------------------------------------------------------------------
    avg = stream->object[ObjID].frame[pos-1].x + stream->object[ObjID].frame[pos].x;
    stream->object[ObjID].frame[pos-1].x = avg / 2;

    avg = stream->object[ObjID].frame[pos-1].y + stream->object[ObjID].frame[pos].y;
    stream->object[ObjID].frame[pos-1].y = avg / 2;

    avg = stream->object[ObjID].frame[pos-1].z + stream->object[ObjID].frame[pos].z;
    stream->object[ObjID].frame[pos-1].z = avg / 2;
    //------------------------------------------------------------------------------------


    //------------------------------------------------------------------------------------
    avg = stream->object[ObjID].frame[pos-1].rot1 + stream->object[ObjID].frame[pos].rot1;
    stream->object[ObjID].frame[pos-1].rot1 = avg / 2;

    avg = stream->object[ObjID].frame[pos-1].rot2 + stream->object[ObjID].frame[pos].rot2;
    stream->object[ObjID].frame[pos-1].rot2 = avg / 2;

    avg = stream->object[ObjID].frame[pos-1].rot3 + stream->object[ObjID].frame[pos].rot3;
    stream->object[ObjID].frame[pos-1].rot3 = avg / 2;

    avg = stream->object[ObjID].frame[pos-1].rot4 + stream->object[ObjID].frame[pos].rot4;
    stream->object[ObjID].frame[pos-1].rot4 = avg / 2;
    //------------------------------------------------------------------------------------
  }
 return 1;
}

int smoothTrajectories(struct VirtualStream * stream)
{
  fprintf(stderr,"Smoothing %u objects \n",stream->numberOfObjects);
  unsigned int objID=0;
  for (objID=0; objID<stream->numberOfObjects; objID++)
  {
     smoothTrajectoriesOfObject(stream,objID);
  }
 return 1;
}



float calculateDistanceTra(float from_x,float from_y,float from_z,float to_x,float to_y,float to_z)
{
   float vect_x = from_x - to_x;
   float vect_y = from_y - to_y;
   float vect_z = from_z - to_z;

   return  (sqrt(pow(vect_x, 2) + pow(vect_y, 2) + pow(vect_z, 2)));

}

void euler2QuaternionsInternal(float * quaternions,float * euler,int quaternionConvention)
{
  #warning "TODO : make this use euler2Quaternions declared at AmMatrix/quaternions.h"
  //This conversion follows the rule euler X Y Z  to quaternions W X Y Z
  //Our input is degrees so we convert it to radians for the sin/cos functions
  float eX = (float) (euler[0] * PI) / 180;
  float eY = (float) (euler[1] * PI) / 180;
  float eZ = (float) (euler[2] * PI) / 180;

  //fprintf(stderr,"eX %f eY %f eZ %f\n",eX,eY,eZ);

  //http://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
  //eX Roll  φ - rotation about the X-axis
  //eY Pitch θ - rotation about the Y-axis
  //eZ Yaw   ψ - rotation about the Z-axis

  float cosX2 = cos((float) eX/2); //cos(φ/2);
  float sinX2 = sin((float) eX/2); //sin(φ/2);
  float cosY2 = cos((float) eY/2); //cos(θ/2);
  float sinY2 = sin((float) eY/2); //sin(θ/2);
  float cosZ2 = cos((float) eZ/2); //cos(ψ/2);
  float sinZ2 = sin((float) eZ/2); //sin(ψ/2);

  switch (quaternionConvention )
  {
   case 1 :
   /*qX*/ quaternions[0] = (sinX2 * cosY2 * cosZ2) - (cosX2 * sinY2 * sinZ2);
   /*qY*/ quaternions[1] = (cosX2 * sinY2 * cosZ2) + (sinX2 * cosY2 * sinZ2);
   /*qZ*/ quaternions[2] = (cosX2 * cosY2 * sinZ2) - (sinX2 * sinY2 * cosZ2);
   /*qW*/ quaternions[3] = (cosX2 * cosY2 * cosZ2) + (sinX2 * sinY2 * sinZ2);
   break;

   case 0 :
   /*qW*/ quaternions[0] = (cosX2 * cosY2 * cosZ2) + (sinX2 * sinY2 * sinZ2);
   /*qX*/ quaternions[1] = (sinX2 * cosY2 * cosZ2) - (cosX2 * sinY2 * sinZ2);
   /*qY*/ quaternions[2] = (cosX2 * sinY2 * cosZ2) + (sinX2 * cosY2 * sinZ2);
   /*qZ*/ quaternions[3] = (cosX2 * cosY2 * sinZ2) - (sinX2 * sinY2 * cosZ2);
   break;

   default :
    fprintf(stderr,"Unhandled quaternion order given (%u) \n",quaternionConvention);
   break;
  };

}

int convertQuaternionsToEulerAngles(struct VirtualStream * stream,float * euler,float *quaternion)
{
 normalizeQuaternions(&quaternion[0],&quaternion[1],&quaternion[2],&quaternion[3]);
 float eulerTMP[3];
 quaternions2Euler(eulerTMP,quaternion,1); //1
 euler[0] = stream->rotationsOffset[0] + (stream->scaleWorld[3] * eulerTMP[0]);
 euler[1] = stream->rotationsOffset[1] + (stream->scaleWorld[4] * eulerTMP[1]);
 euler[2] = stream->rotationsOffset[2] + (stream->scaleWorld[5] * eulerTMP[2]);

 #if PRINT_LOAD_INFO
                fprintf(stderr,"Tracker OBJ%u( %f %f %f ,  %f %f %f )\n",item,pos[0],pos[1],pos[2],pos[3],pos[4],pos[5]);
                fprintf(stderr,"Angle Offset %f %f %f \n",newstream->rotationsOffset[0],newstream->rotationsOffset[1],newstream->rotationsOffset[2]);
 #endif


  if (stream->rotationsOverride)
    { flipRotationAxisD(&euler[0],&euler[1],&euler[2], stream->rotationsXYZ[0] , stream->rotationsXYZ[1] , stream->rotationsXYZ[2]); }

 return 1;
}

int parseAbsoluteRotation(struct VirtualStream * stream,float * planetRotAbsolute,unsigned int planetObj,unsigned int frameNumber)
{
   if (stream->object[planetObj].frame[frameNumber].isQuaternion)
      {
        float euler[3];
        float quaternions[4];
        quaternions[0]=stream->object[planetObj].frame[frameNumber].rot1;
        quaternions[1]=stream->object[planetObj].frame[frameNumber].rot2;
        quaternions[2]=stream->object[planetObj].frame[frameNumber].rot3;
        quaternions[3]=stream->object[planetObj].frame[frameNumber].rot4;
        convertQuaternionsToEulerAngles(stream,euler,quaternions);
        planetRotAbsolute[0] = euler[0];
        planetRotAbsolute[1] = euler[1];
        planetRotAbsolute[2] = euler[2];
      } else
      {
       planetRotAbsolute[0] = (float) stream->object[planetObj].frame[frameNumber].rot1;
       planetRotAbsolute[1] = (float) stream->object[planetObj].frame[frameNumber].rot2;
       planetRotAbsolute[2] = (float) stream->object[planetObj].frame[frameNumber].rot3;
      }
  return 1;
}


int affixSatteliteToPlanetFromFrameForLength(struct VirtualStream * stream,unsigned int satteliteObj,unsigned int planetObj , unsigned int frameNumber , unsigned int duration)
{
    fprintf(stderr,"affixSatteliteToPlanetFromFrameForLength disabled after float transition\n");
    return 0;
    /*
  fprintf(stderr,"affixSatteliteToPlanetFromFrameForLength(sat=%u,planet=%u) from frame %u to frame %u \n",satteliteObj,planetObj,frameNumber,frameNumber+duration);
  if ( satteliteObj >= stream->numberOfObjects ) { fprintf(stderr,RED "affixSatteliteToPlanetFromFrameForLength referencing non existent Object %u\n" NORMAL,satteliteObj); return 0; }
  if ( planetObj >= stream->numberOfObjects )    { fprintf(stderr,RED "affixSatteliteToPlanetFromFrameForLength referencing non existent Object %u\n" NORMAL,planetObj);    return 0; }

  unsigned int satteliteObjFrameNumber = stream->object[satteliteObj].numberOfFrames;
  unsigned int planetObjFrameNumber    = stream->object[planetObj].numberOfFrames;

  unsigned int satteliteObjMaxFrameNumber = stream->object[satteliteObj].MAX_numberOfFrames;
  unsigned int planetObjMaxFrameNumber    = stream->object[planetObj].MAX_numberOfFrames;

  if ( satteliteObjMaxFrameNumber <  planetObjFrameNumber )
     {
       unsigned int growthSize = planetObjFrameNumber-satteliteObjMaxFrameNumber+frameNumber+1;
       fprintf(stderr,GREEN" sattelite : growing sattelite stream to accomodate %u poses , as many as the planet stream\n" NORMAL,growthSize);
       //GROW STREAM HERE
       growVirtualStreamFrames(&stream->object[satteliteObj],growthSize);
       satteliteObjFrameNumber = stream->object[satteliteObj].numberOfFrames;
       satteliteObjMaxFrameNumber = stream->object[satteliteObj].MAX_numberOfFrames;
     }



  if ( satteliteObjFrameNumber <= frameNumber )
     {
       fprintf(stderr,RED " sattelite : referencing non existent frames ( %u ) \n" NORMAL,frameNumber);
       return 0;
     }
  if ( satteliteObjMaxFrameNumber < frameNumber+duration )
     {
       fprintf(stderr,RED " sattelite : referencing non existent frames ( want %u + %u frames , but max frame is %u ) \n" NORMAL,frameNumber,duration,satteliteObjMaxFrameNumber);
       duration = satteliteObjMaxFrameNumber-frameNumber;
       fprintf(stderr,RED " sattelite : correcting duration to %u\n" NORMAL,duration);
     }

  // PLANET CODE --------------------------------------------------------------
  if ( planetObjFrameNumber <= frameNumber )
     {
       fprintf(stderr,RED " planet : referencing non existent frames ( %u ) \n" NORMAL,frameNumber);
       return 0;
     }
  if ( planetObjFrameNumber < frameNumber+duration )
     {
       fprintf(stderr,RED " planet : referencing non existent frames ( want %u + %u frames , but max frame is %u ) \n" NORMAL,frameNumber,duration,planetObjFrameNumber);
       duration = planetObjFrameNumber-frameNumber;
       fprintf(stderr,RED " planet : correcting duration to %u\n" NORMAL,duration);
     }

    //There is literally no good reason to go from rotation -> quaternion -> 3x3 -> quaternion -> rotation this could be optimized
    //==================================================================================
    float satPosAbsolute[4]={0};
    satPosAbsolute[0] = (float) stream->object[satteliteObj].frame[frameNumber].x;
    satPosAbsolute[1] = (float) stream->object[satteliteObj].frame[frameNumber].y;
    satPosAbsolute[2] = (float) stream->object[satteliteObj].frame[frameNumber].z;
    satPosAbsolute[3] = 1.0;
    //==================================================================================
    float planetPosAbsolute[4]={0};
    planetPosAbsolute[0] = (float) stream->object[planetObj].frame[frameNumber].x;
    planetPosAbsolute[1] = (float) stream->object[planetObj].frame[frameNumber].y;
    planetPosAbsolute[2] = (float) stream->object[planetObj].frame[frameNumber].z;
    planetPosAbsolute[3] = 1.0;

    float planetRotAbsolute[4]={0};
    planetRotAbsolute[0] = (float) stream->object[planetObj].frame[frameNumber].rot1;
    planetRotAbsolute[1] = (float) stream->object[planetObj].frame[frameNumber].rot2;
    planetRotAbsolute[2] = (float) stream->object[planetObj].frame[frameNumber].rot3;


    parseAbsoluteRotation(stream,planetRotAbsolute,planetObj,frameNumber);
    //==================================================================================


    float satPosRelative[4]={0};
    pointFromAbsoluteToRelationWithObject_PosXYZRotationXYZ(satPosRelative,planetPosAbsolute,planetRotAbsolute,satPosAbsolute);

    unsigned int pos=0;
    fprintf(stderr,YELLOW " Will align satelite to planet from frame %u to %u\n" NORMAL ,frameNumber+1 , frameNumber+duration );
    for (pos=frameNumber+1; pos<frameNumber+duration; pos++)
    {
       planetPosAbsolute[0] = (float) stream->object[planetObj].frame[pos].x;
       planetPosAbsolute[1] = (float) stream->object[planetObj].frame[pos].y;
       planetPosAbsolute[2] = (float) stream->object[planetObj].frame[pos].z;
       planetPosAbsolute[3] = 1.0;

       parseAbsoluteRotation(stream,planetRotAbsolute,planetObj,frameNumber);

       if ( pointFromRelationWithObjectToAbsolute_PosXYZRotationXYZ(satPosAbsolute,planetPosAbsolute,planetRotAbsolute,satPosRelative) )
       {
         stream->object[satteliteObj].frame[pos].x = (float) satPosAbsolute[0];
         stream->object[satteliteObj].frame[pos].y = (float) satPosAbsolute[1];
         stream->object[satteliteObj].frame[pos].z = (float) satPosAbsolute[2];
         stream->object[satteliteObj].frame[pos].time = stream->object[planetObj].frame[pos].time;
         //stream->object[satteliteObj].frame[pos].isQuaternion = 0;
       }

       //  stream->object[satteliteObj].frame[pos].x = stream->object[planetObj].frame[pos].x;
      //   stream->object[satteliteObj].frame[pos].y = stream->object[planetObj].frame[pos].y;
      //   stream->object[satteliteObj].frame[pos].z = stream->object[planetObj].frame[pos].z-0.5;
    }

    //Everything is set now to mark the sattelite new end
    stream->object[satteliteObj].numberOfFrames = stream->object[planetObj].numberOfFrames;// frameNumber+duration; //stream->object[planetObj].numberOfFrames;
    stream->object[satteliteObj].lastFrame = stream->object[planetObj].lastFrame;
    stream->object[satteliteObj].MAX_timeOfFrames = stream->object[planetObj].MAX_timeOfFrames;
    stream->object[satteliteObj].numberOfFrames = stream->object[planetObj].numberOfFrames;
 return 1;
*/
}


int objectsCollide(struct VirtualStream * newstream,unsigned int atTime,unsigned int objIDA,unsigned int objIDB)
{
  float posA[7]={0}; float scaleA_X,scaleA_Y,scaleA_Z;
  float posB[7]={0}; float scaleB_X,scaleB_Y,scaleB_Z;

  calculateVirtualStreamPos(newstream,objIDA,atTime,posA,0,&scaleA_X,&scaleA_Y,&scaleA_Z);
  calculateVirtualStreamPos(newstream,objIDB,atTime,posB,0,&scaleB_X,&scaleB_Y,&scaleB_Z);

  float distance =  calculateDistanceTra(posA[0],posA[1],posA[2],posB[0],posB[1],posB[2]);
  fprintf(stderr,"Distance %u from %u = %f\n",objIDA,objIDB,distance);
  if ( distance > 0.3 ) { return 0;}

  return 1;
}

int flipRotationAxisD(float * rotX, float * rotY , float * rotZ , int where2SendX , int where2SendY , int where2SendZ)
{
  #if PRINT_LOAD_INFO
   fprintf(stderr,"Had rotX %f rotY %f rotZ %f \n",*rotX,*rotY,*rotZ);
   fprintf(stderr,"Moving 0 to %u , 1 to %u , 2 to %u \n",where2SendX,where2SendY,where2SendZ);
  #endif

  float tmpX = *rotX;
  float tmpY = *rotY;
  float tmpZ = *rotZ;
  //-----------------------------------------
  if (where2SendX==0) { *rotX=tmpX; } else
  if (where2SendX==1) { *rotY=tmpX; } else
  if (where2SendX==2) { *rotZ=tmpX; }

  if (where2SendY==0) { *rotX=tmpY; } else
  if (where2SendY==1) { *rotY=tmpY; } else
  if (where2SendY==2) { *rotZ=tmpY; }

  if (where2SendZ==0) { *rotX=tmpZ; } else
  if (where2SendZ==1) { *rotY=tmpZ; } else
  if (where2SendZ==2) { *rotZ=tmpZ; }
  //-----------------------------------------

  #if PRINT_LOAD_INFO
   fprintf(stderr,"Now have rotX %f rotY %f rotZ %f \n",*rotX,*rotY,*rotZ);
  #endif

  return 1;
}

int flipRotationAxis(float * rotX, float * rotY , float * rotZ , int where2SendX , int where2SendY , int where2SendZ)
{
  #if PRINT_LOAD_INFO
   fprintf(stderr,"Had rotX %f rotY %f rotZ %f \n",*rotX,*rotY,*rotZ);
   fprintf(stderr,"Moving 0 to %u , 1 to %u , 2 to %u \n",where2SendX,where2SendY,where2SendZ);
  #endif

  float tmpX = *rotX;
  float tmpY = *rotY;
  float tmpZ = *rotZ;
  //-----------------------------------------
  if (where2SendX==0) { *rotX=tmpX; } else
  if (where2SendX==1) { *rotY=tmpX; } else
  if (where2SendX==2) { *rotZ=tmpX; }

  if (where2SendY==0) { *rotX=tmpY; } else
  if (where2SendY==1) { *rotY=tmpY; } else
  if (where2SendY==2) { *rotZ=tmpY; }

  if (where2SendZ==0) { *rotX=tmpZ; } else
  if (where2SendZ==1) { *rotY=tmpZ; } else
  if (where2SendZ==2) { *rotZ=tmpZ; }
  //-----------------------------------------

  #if PRINT_LOAD_INFO
   fprintf(stderr,"Now have rotX %f rotY %f rotZ %f \n",*rotX,*rotY,*rotZ);
  #endif

  return 1;
}



int unflipRotationAxis(float * rotX, float * rotY , float * rotZ , int where2SendX , int where2SendY , int where2SendZ)
{
  #warning "Is   unflipRotationAxis correct ? "
  #if PRINT_LOAD_INFO
   fprintf(stderr,"Had rotX %f rotY %f rotZ %f \n",*rotX,*rotY,*rotZ);
   fprintf(stderr,"Moving 0 to %u , 1 to %u , 2 to %u \n",where2SendX,where2SendY,where2SendZ);
  #endif

  float tmpX = 0;
  float tmpY = 0;
  float tmpZ = 0;
  //-----------------------------------------
  if (where2SendX==0) { tmpX=*rotX; } else
  if (where2SendX==1) { tmpX=*rotY; } else
  if (where2SendX==2) { tmpX=*rotZ; }

  if (where2SendY==0) { tmpY=*rotX; } else
  if (where2SendY==1) { tmpY=*rotY; } else
  if (where2SendY==2) { tmpY=*rotZ; }

  if (where2SendZ==0) { tmpZ=*rotX; } else
  if (where2SendZ==1) { tmpZ=*rotY; } else
  if (where2SendZ==2) { tmpZ=*rotZ; }
  //-----------------------------------------

  *rotX=tmpX;
  *rotY=tmpY;
  *rotZ=tmpZ;

  #if PRINT_LOAD_INFO
   fprintf(stderr,"Now have rotX %f rotY %f rotZ %f \n",*rotX,*rotY,*rotZ);
  #endif

  return 1;
}



int fillPosWithNull(float * pos,float * scaleX ,float * scaleY,float * scaleZ)
{
    #if PRINT_DEBUGGING_INFO
    fprintf(stderr,"Returning null frame for obj %u \n",ObjID);
    #endif

    pos[0]=0.0;
    pos[1]=0.0;
    pos[2]=0.0;
    pos[3]=0.0;
    pos[4]=0.0;
    pos[5]=0.0;
    pos[6]=0.0;
    *scaleX = 1.0;
    *scaleY = 1.0;
    *scaleZ = 1.0;

    return 1;
}


int fillPosWithLastFrame(
                          struct VirtualStream * stream,
                          ObjectIDHandler ObjID,
                          float * pos,
                          float * joints,
                          float * scaleX,
                          float * scaleY,
                          float * scaleZ
                        )
{
   if (stream==0) { fprintf(stderr,"Cannot fill position on empty stream \n"); return 0; }
   if (pos==0) { fprintf(stderr,"Cannot fill position on empty position \n"); return 0; }
   if (ObjID>=stream->numberOfObjects) { fprintf(stderr,"Trying to add position for a non existing object\n"); return 0; }

   if (stream->object[ObjID].frame==0)
    {
      #if PRINT_WARNING_INFO
       fprintf(stderr,"Cannot Access frames for object %u \n",ObjID);
      #endif
      return 0;
    }

    #if PRINT_DEBUGGING_INFO
    fprintf(stderr,"Returning frame %u \n",FrameIDToReturn);
    #endif

    unsigned int FrameIDToReturn = stream->object[ObjID].numberOfFrames;
    if (FrameIDToReturn>0) { --FrameIDToReturn; } //We have FrameIDToReturn frames so we grab the last one ( FrameIDToReturn -1 )
    pos[0]=stream->object[ObjID].frame[FrameIDToReturn].x;
    pos[1]=stream->object[ObjID].frame[FrameIDToReturn].y;
    pos[2]=stream->object[ObjID].frame[FrameIDToReturn].z;
    pos[3]=stream->object[ObjID].frame[FrameIDToReturn].rot1;
    pos[4]=stream->object[ObjID].frame[FrameIDToReturn].rot2;
    pos[5]=stream->object[ObjID].frame[FrameIDToReturn].rot3;
    pos[6]=stream->object[ObjID].frame[FrameIDToReturn].rot4;
    *scaleX=stream->object[ObjID].frame[FrameIDToReturn].scaleX;
    *scaleY=stream->object[ObjID].frame[FrameIDToReturn].scaleY;
    *scaleZ=stream->object[ObjID].frame[FrameIDToReturn].scaleZ;
    return 1;
}



int fillPosWithFrame(
                      struct VirtualStream * stream,
                      ObjectIDHandler ObjID,
                      unsigned int FrameIDToReturn,
                      float * pos,
                      float * joints,
                      float * scaleX,
                      float * scaleY,
                      float * scaleZ
                    )
{
   if (stream->object[ObjID].frame==0)
    {
      #if PRINT_WARNING_INFO
      fprintf(stderr,"Cannot Access frames for object %u \n",ObjID);
      #endif
      return 0;
    }

    #if PRINT_DEBUGGING_INFO
    fprintf(stderr,"Returning frame %u \n",FrameIDToReturn);
    #endif

    if (FrameIDToReturn >= stream->object[ObjID].MAX_numberOfFrames )
     {
         fprintf(
                 stderr,"fillPosWithFrame asked to return frame out of bounds ( %u / %u / %u Max ) \n",
                 FrameIDToReturn,
                 stream->object[ObjID].numberOfFrames,
                 stream->object[ObjID].MAX_numberOfFrames
                );
         return 0;
     }

    pos[0]=stream->object[ObjID].frame[FrameIDToReturn].x;
    pos[1]=stream->object[ObjID].frame[FrameIDToReturn].y;
    pos[2]=stream->object[ObjID].frame[FrameIDToReturn].z;
    pos[3]=stream->object[ObjID].frame[FrameIDToReturn].rot1;
    pos[4]=stream->object[ObjID].frame[FrameIDToReturn].rot2;
    pos[5]=stream->object[ObjID].frame[FrameIDToReturn].rot3;
    pos[6]=stream->object[ObjID].frame[FrameIDToReturn].rot4;
    *scaleX=stream->object[ObjID].frame[FrameIDToReturn].scaleX;
    *scaleY=stream->object[ObjID].frame[FrameIDToReturn].scaleY;
    *scaleZ=stream->object[ObjID].frame[FrameIDToReturn].scaleZ;


    if ( (joints!=0) && (stream->object[ObjID].frame[FrameIDToReturn].jointList!=0) )
    {
      /*
       fprintf(stderr,
               "Populating non interpolated joints for frame %u  ( %u joints ) \n",
               FrameIDToReturn ,
               stream->object[ObjID].frame[FrameIDToReturn].jointList->numberOfJoints
              );
       */
       unsigned int numberOfJoints = stream->object[ObjID].frame[FrameIDToReturn].jointList->numberOfJoints;

       struct Matrix4x4OfFloats mF={0};
         
       float rotCur[4]={0};
       unsigned int i=0,z=0;
       for (i=0; i<numberOfJoints; i++)
       {
        float * f=&joints[16*i]; 

        //Euler Rotation
        //---------------------------------------------------------------------------------------------------------------------------------
        if (stream->object[ObjID].frame[FrameIDToReturn].jointList->joint[i].useEulerRotation)
        {
          if (!stream->object[ObjID].frame[FrameIDToReturn].jointList->joint[i].eulerRotationOrder)
          {
              fprintf(stderr,RED "fillPosWithFrame: Empty eulerRotationOrder %u #%u/%u \n" NORMAL,FrameIDToReturn,i,numberOfJoints);
          }

         rotCur[0] = stream->object[ObjID].frame[FrameIDToReturn].jointList->joint[i].rot1;
         rotCur[1] = stream->object[ObjID].frame[FrameIDToReturn].jointList->joint[i].rot2;
         rotCur[2] = stream->object[ObjID].frame[FrameIDToReturn].jointList->joint[i].rot3;
         rotCur[3] = 0.0;
         
         struct Matrix4x4OfFloats mF={0};

         create4x4FMatrixFromEulerAnglesWithRotationOrder(
                                                          &mF,
                                                          rotCur[0],
                                                          rotCur[1],
                                                          rotCur[2],
                                                          (unsigned int) stream->object[ObjID].frame[FrameIDToReturn].jointList->joint[i].eulerRotationOrder
                                                        );

         copy4x4FMatrix(f,mF.m);
        } else
        //Quaternion Rotation
        //---------------------------------------------------------------------------------------------------------------------------------
        if (stream->object[ObjID].frame[FrameIDToReturn].jointList->joint[i].useQuaternion)
        {
         rotCur[0] = stream->object[ObjID].frame[FrameIDToReturn].jointList->joint[i].rot1;
         rotCur[1] = stream->object[ObjID].frame[FrameIDToReturn].jointList->joint[i].rot2;
         rotCur[2] = stream->object[ObjID].frame[FrameIDToReturn].jointList->joint[i].rot3;
         rotCur[3] = stream->object[ObjID].frame[FrameIDToReturn].jointList->joint[i].rot4;
          
         create4x4FQuaternionMatrix(&mF,rotCur[0],rotCur[1],rotCur[2],rotCur[3]); 
         //quaternion2Matrix4x4(m,rotCur,0);
         copy4x4FMatrix(f,mF.m);
        } else
        //Matrix 4x4 Matrix
        //---------------------------------------------------------------------------------------------------------------------------------
        if (stream->object[ObjID].frame[FrameIDToReturn].jointList->joint[i].useMatrix4x4)
        {
          //If we want to use a 4x4 matrix then just copy it..
          copy4x4FMatrix(
                          f,
                          stream->object[ObjID].frame[FrameIDToReturn].jointList->joint[i].m
                        );
          /*
          float *s = stream->object[ObjID].frame[FrameIDToReturn].jointList->joint[i].m;
          for (z=0; z<16; z++)
            {
              f[z]=s[z];
            }*/
        } else
        //Not used so identity 4x4 Matrix
        //---------------------------------------------------------------------------------------------------------------------------------
        {
         //fprintf(stderr,"fillPosWithFrame: Empty Joint -> Identity Matrix %u #%u/%u \n",FrameIDToReturn,i,numberOfJoints);
         create4x4FIdentityMatrix(&mF);
         copy4x4FMatrix(f,mF.m);
        }
        //---------------------------------------------------------------------------------------------------------------------------------
       }
    }

    return 1;
}



int fillJointsWithInterpolatedFrame(
                                  struct VirtualStream * stream,
                                  ObjectIDHandler ObjID,
                                  float * joints,
                                  unsigned int PrevFrame,
                                  unsigned int NextFrame ,
                                  unsigned int our_stepTime,
                                  unsigned int MAX_stepTime
                                )
{
//This call will not work correctly if we mix 4x4 matrices , eulers and quaternions on prevFrame/nextFrame

       float rotPrev[4]={0};
       float rotNext[4]={0};
       float rotTot[4]={0};
       unsigned int numberOfJoints = 0;

       if (stream->object[ObjID].frame[PrevFrame].jointList!=0) { numberOfJoints = stream->object[ObjID].frame[PrevFrame].jointList->numberOfJoints; }
       if (numberOfJoints==0) { return 1; }

       if (stream->object[ObjID].frame[NextFrame].jointList!=0) { numberOfJoints = stream->object[ObjID].frame[NextFrame].jointList->numberOfJoints; }
       if (numberOfJoints==0) { return 1; }


       if (MAX_stepTime==0)   { return 0; }

       float timeMultiplier = (float) our_stepTime / MAX_stepTime;

       unsigned int i=0;
       for (i=0; i<numberOfJoints; i++)
       {

        float * f=&joints[16*i]; 

        if (
            (stream->object[ObjID].frame[PrevFrame].jointList==0) ||
            (stream->object[ObjID].frame[NextFrame].jointList==0)
           )
         {
           // No joints declared so impossible to interpolate
           // so doing nothing , dont print something because it will slow execution
         } else
        if (
            (
             (stream->object[ObjID].frame[PrevFrame].jointList->joint[i].useEulerRotation) ||
             (stream->object[ObjID].frame[PrevFrame].jointList->joint[i].useQuaternion)
            ) ||
            (
             (stream->object[ObjID].frame[NextFrame].jointList->joint[i].useEulerRotation) ||
             (stream->object[ObjID].frame[NextFrame].jointList->joint[i].useQuaternion)
            )
           )
        {
         if (stream->object[ObjID].frame[PrevFrame].hasNonDefaultJointList)
         {
          rotPrev[0] = stream->object[ObjID].frame[PrevFrame].jointList->joint[i].rot1;
          rotPrev[1] = stream->object[ObjID].frame[PrevFrame].jointList->joint[i].rot2;
          rotPrev[2] = stream->object[ObjID].frame[PrevFrame].jointList->joint[i].rot3;
          rotPrev[3] = stream->object[ObjID].frame[PrevFrame].jointList->joint[i].rot4;
         }

         if (stream->object[ObjID].frame[NextFrame].hasNonDefaultJointList)
         {
          rotNext[0] = stream->object[ObjID].frame[NextFrame].jointList->joint[i].rot1;
          rotNext[1] = stream->object[ObjID].frame[NextFrame].jointList->joint[i].rot2;
          rotNext[2] = stream->object[ObjID].frame[NextFrame].jointList->joint[i].rot3;
          rotNext[3] = stream->object[ObjID].frame[NextFrame].jointList->joint[i].rot4;
         }


         rotTot[0] = rotPrev[0] + (float) ( rotNext[0] - rotPrev[0] ) * timeMultiplier;
         rotTot[1] = rotPrev[1] + (float) ( rotNext[1] - rotPrev[1] ) * timeMultiplier;
         rotTot[2] = rotPrev[2] + (float) ( rotNext[2] - rotPrev[2] ) * timeMultiplier;
         rotTot[3] = rotPrev[3] + (float) ( rotNext[3] - rotPrev[3] ) * timeMultiplier;

         //fprintf(stderr,"Rotation Prev (obj=%u pos=%u bone=%u ) is %0.2f %0.2f %0.2f \n",ObjID,PrevFrame,i,rotPrev[0],rotPrev[1],rotPrev[2]);
         //fprintf(stderr,"Rotation Next (obj=%u pos=%u bone=%u ) is %0.2f %0.2f %0.2f \n",ObjID,NextFrame,i,rotNext[0],rotNext[1],rotNext[2]);
         //fprintf(stderr,"Rotation Requested  is %0.2f %0.2f %0.2f ( mult %0.2f ) \n",rotTot[0],rotTot[1],rotTot[2],timeMultiplier);
         
         struct Matrix4x4OfFloats mF={0};
         
         create4x4FMatrixFromEulerAnglesWithRotationOrder(
                                                          &mF,
                                                          (float) rotTot[0],
                                                          (float) rotTot[1],
                                                          (float) rotTot[2],
                                                          (unsigned int) stream->object[ObjID].frame[NextFrame].jointList->joint[i].eulerRotationOrder
                                                        );
         //create4x4MatrixFromEulerAnglesXYZ(m,rotTot[0],rotTot[1],rotTot[2]);
         copy4x4FMatrix(f,mF.m);
        } else
       if (
             (stream->object[ObjID].frame[PrevFrame].jointList->joint[i].useMatrix4x4)
             ||
             (stream->object[ObjID].frame[NextFrame].jointList->joint[i].useMatrix4x4)
          )
        {
        //fprintf(stderr,"Rotation MAT4x4 (obj=%u pos=%u bone=%u ) \n",ObjID,PrevFrame,i);
        slerp2RotTransMatrices4x4(
                                   f , //write straight to the output
                                   stream->object[ObjID].frame[PrevFrame].jointList->joint[i].m,
                                   stream->object[ObjID].frame[NextFrame].jointList->joint[i].m ,
                                   timeMultiplier
                                  );
        } else
        {
          if (
              (stream->object[ObjID].frame[PrevFrame].jointList->joint[i].altered) ||
              (stream->object[ObjID].frame[NextFrame].jointList->joint[i].altered)
             )
             {
              fprintf(stderr,RED "Unknown interpolation combination  ( obj %u , joint %u ) only supporting Euler->Euler , M4x4->M4x4\n" NORMAL , ObjID , i);
              fprintf(stderr,"PrevFrame ( %u ) : \n",PrevFrame);
              fprintf(stderr," euler  switch  : %u \n",stream->object[ObjID].frame[PrevFrame].jointList->joint[i].useEulerRotation);
              fprintf(stderr," quat   switch  : %u \n",stream->object[ObjID].frame[PrevFrame].jointList->joint[i].useQuaternion);
              fprintf(stderr," mat4x4 switch  : %u \n",stream->object[ObjID].frame[PrevFrame].jointList->joint[i].useMatrix4x4);

              fprintf(stderr,"NextFrame ( %u ): \n",NextFrame);
              fprintf(stderr," euler  switch  : %u \n",stream->object[ObjID].frame[NextFrame].jointList->joint[i].useEulerRotation);
              fprintf(stderr," quat   switch  : %u \n",stream->object[ObjID].frame[NextFrame].jointList->joint[i].useQuaternion);
              fprintf(stderr," mat4x4 switch  : %u \n",stream->object[ObjID].frame[NextFrame].jointList->joint[i].useMatrix4x4);
             } else
             {
               //No information set , we shouldnt even be trying for an interpolation
             }

        }
      }
  return 1;
}









int fillPosWithInterpolatedFrame(
                                  struct VirtualStream * stream,
                                  ObjectIDHandler ObjID,
                                  float * pos,
                                  float * joints,
                                  float * scaleX,
                                  float * scaleY,
                                  float * scaleZ,
                                  unsigned int PrevFrame,
                                  unsigned int NextFrame ,
                                  unsigned int time
                                )
{
   if (stream->object[ObjID].frame==0)
    {
      #if PRINT_WARNING_INFO
      fprintf(stderr,"Cannot Access interpolated frames for object %u \n",ObjID);
      #endif
      return 0;
    }

   if (PrevFrame==NextFrame)
    {
       return fillPosWithFrame(
                                stream,
                                ObjID,
                                PrevFrame,
                                pos,
                                joints,
                                scaleX,
                                scaleY,
                                scaleZ
                              );
    }


    #if PRINT_DEBUGGING_INFO
    fprintf(stderr,"Interpolating frames @  %u , between %u and %u \n",time,PrevFrame,NextFrame);
    #endif
    float interPos[7]={0};
    float interScale;

    unsigned int MAX_stepTime= stream->object[ObjID].frame[NextFrame].time - stream->object[ObjID].frame[PrevFrame].time;
    if (MAX_stepTime == 0 ) { MAX_stepTime=1; }
    unsigned int our_stepTime= time - stream->object[ObjID].frame[PrevFrame].time;


    interPos[0]=(float) ( stream->object[ObjID].frame[NextFrame].x-stream->object[ObjID].frame[PrevFrame].x ) * our_stepTime / MAX_stepTime;
    interPos[0]+=stream->object[ObjID].frame[PrevFrame].x;

    interPos[1]=(float) ( stream->object[ObjID].frame[NextFrame].y-stream->object[ObjID].frame[PrevFrame].y ) * our_stepTime / MAX_stepTime;
    interPos[1]+=stream->object[ObjID].frame[PrevFrame].y;

    interPos[2]=(float) ( stream->object[ObjID].frame[NextFrame].z-stream->object[ObjID].frame[PrevFrame].z ) * our_stepTime / MAX_stepTime;
    interPos[2]+=stream->object[ObjID].frame[PrevFrame].z;

    interPos[3]=(float) ( stream->object[ObjID].frame[NextFrame].rot1-stream->object[ObjID].frame[PrevFrame].rot1 ) * our_stepTime / MAX_stepTime;
    interPos[3]+=stream->object[ObjID].frame[PrevFrame].rot1;

    interPos[4]=(float) ( stream->object[ObjID].frame[NextFrame].rot2-stream->object[ObjID].frame[PrevFrame].rot2 ) * our_stepTime / MAX_stepTime;
    interPos[4]+=stream->object[ObjID].frame[PrevFrame].rot2;

    interPos[5]=(float) ( stream->object[ObjID].frame[NextFrame].rot3-stream->object[ObjID].frame[PrevFrame].rot3 ) * our_stepTime / MAX_stepTime;
    interPos[5]+=stream->object[ObjID].frame[PrevFrame].rot3;

    interPos[6]=(float) ( stream->object[ObjID].frame[NextFrame].rot4-stream->object[ObjID].frame[PrevFrame].rot4 ) * our_stepTime / MAX_stepTime;
    interPos[6]+=stream->object[ObjID].frame[PrevFrame].rot4;

    interScale = (float) ( stream->object[ObjID].frame[NextFrame].scaleX -stream->object[ObjID].frame[PrevFrame].scaleX ) * our_stepTime / MAX_stepTime;
    interScale += stream->object[ObjID].frame[PrevFrame].scaleX;
    *scaleX=interScale;

    interScale = (float) ( stream->object[ObjID].frame[NextFrame].scaleY -stream->object[ObjID].frame[PrevFrame].scaleY ) * our_stepTime / MAX_stepTime;
    interScale += stream->object[ObjID].frame[PrevFrame].scaleY;
    *scaleY=interScale;

    interScale = (float) ( stream->object[ObjID].frame[NextFrame].scaleZ -stream->object[ObjID].frame[PrevFrame].scaleZ ) * our_stepTime / MAX_stepTime;
    interScale += stream->object[ObjID].frame[PrevFrame].scaleZ;
    *scaleZ=interScale;


    pos[0]=interPos[0]; pos[1]=interPos[1]; pos[2]=interPos[2];
    pos[3]=interPos[3]; pos[4]=interPos[4]; pos[5]=interPos[5];
    pos[6]=interPos[6];


    if ( (joints!=0) &&
         (
          (stream->object[ObjID].frame[PrevFrame].jointList!=0) ||
          (stream->object[ObjID].frame[NextFrame].jointList!=0)
         )
        )
    {
       //fprintf(stderr,"interpolating joints for frame %u  ( %u joints / obj %u ) \n",PrevFrame , stream->object[ObjID].frame[PrevFrame].jointList->numberOfJoints , ObjID);
       fillJointsWithInterpolatedFrame(
                                        stream,
                                        ObjID,
                                        joints,
                                        PrevFrame,
                                        NextFrame ,
                                        our_stepTime,
                                        MAX_stepTime
                                       );
    }

    #if PRINT_DEBUGGING_INFO
    fprintf(stderr,"ok \n");
    #endif

    return 1;
}


int getExactStreamPosFromTimestamp(struct VirtualStream * stream,ObjectIDHandler ObjID,unsigned int timeAbsMilliseconds , int * foundExactTimestamp)
{
  if (foundExactTimestamp==0) { fprintf(stderr,"getExactStreamPosFromTimestamp called with no place to place output\n"); return 0; }
  *foundExactTimestamp=0;

  //fprintf(stderr,"getExactStreamPosFromTimestamp(obj=%u,time=%u) ",ObjID,timeAbsMilliseconds);

  if (stream==0) { fprintf(stderr,"getExactStreamPosFromTimestamp called with null stream\n"); return 0; }
  if (stream->object==0) { fprintf(stderr,"getExactStreamPosFromTimestamp called with null object array\n"); return 0; }
  if (stream->numberOfObjects<=ObjID) { fprintf(stderr,"getExactStreamPosFromTimestamp ObjID %u is out of bounds (%u)\n",ObjID,stream->numberOfObjects); return 0; }
  if ( (ObjID==0) && (stream->object[ObjID].frame == 0 )  ) { /*Special case with non declared cameras , it is ok , dont spam for every frame..! */ return 0; }
     else
  if (stream->object[ObjID].frame == 0 ) { fprintf(stderr,"getExactStreamPosFromTimestamp ObjID %u does not have a frame array allocated\n",ObjID); return 0; }

  if ( (ObjID==0) && (stream->object[ObjID].numberOfFrames == 0 )  ) { /*Special case with non declared cameras , it is ok , dont spam for every frame..! */ return 0; }
    else
  if (stream->object[ObjID].numberOfFrames == 0 ) { fprintf(stderr,"getExactStreamPosFromTimestamp ObjID %u has 0 frames\n",ObjID); return 0; }



  if (stream->debug)
  {
    fprintf(stderr,"searching positions in %u frames\n",stream->object[ObjID].numberOfFrames);
  }

  unsigned int pos=0;
  for (pos=0; pos<stream->object[ObjID].numberOfFrames; pos++)
  {
    if (timeAbsMilliseconds == stream->object[ObjID].frame[pos].time )
    {
         if (stream->debug)
              { fprintf(stderr,"FOUND!\n"); }
        *foundExactTimestamp=1;
        return pos;
    }
  }


  fprintf(stderr,"getExactStreamPosFromTimestamp(obj=%u,time=%u) not found \n",ObjID,timeAbsMilliseconds);
  //fprintf(stderr,"not found!\n");
 return 0;
}



int calculateVirtualStreamPos(
                               struct VirtualStream * stream,
                               ObjectIDHandler ObjID,
                               unsigned int timeAbsMilliseconds,
                               float * pos,
                               float * joints,
                               float * scaleX,
                               float * scaleY,
                               float * scaleZ
                             )
{
   //-----------------------------------------------------------------------------------------------------------------------------------------------
   //First job, check if everything is ok..!
   //-----------------------------------------------------------------------------------------------------------------------------------------------
   if (stream==0) { fprintf(stderr,"calculateVirtualStreamPos called with null stream\n"); return 0; }
   if (stream->object==0) { fprintf(stderr,"calculateVirtualStreamPos called with null object array\n"); return 0; }
   if (stream->numberOfObjects<=ObjID) { fprintf(stderr,"calculateVirtualStreamPos ObjID %u is out of bounds (%u)\n",ObjID,stream->numberOfObjects); return 0; }

   if ( (ObjID==0) && (stream->object[ObjID].frame == 0 )  ) { /*Special case with non declared cameras , it is ok , dont spam for every frame..! */ return 0; }
     else
   if (stream->object[ObjID].frame == 0 ) { fprintf(stderr,"calculateVirtualStreamPos ObjID %u does not have a frame array allocated\n",ObjID); return 0; }


   if ( (ObjID==0) && (stream->object[ObjID].numberOfFrames == 0 )  ) { /*Special case with non declared cameras , it is ok , dont spam for every frame..! */ return 0; }
    else
   if (stream->object[ObjID].numberOfFrames == 0 ) { fprintf(stderr,"calculateVirtualStreamPos ObjID %u has 0 frames\n",ObjID); return 0; }
   //-----------------------------------------------------------------------------------------------------------------------------------------------

   if (
        (stream->autoRefresh != 0 ) ||
        (stream->autoRefreshForce)
      )
    {
         //Check for refreshed version ?
       if  (
            (stream->autoRefreshForce) ||
            (stream->autoRefresh < timeAbsMilliseconds-stream->lastRefresh )
           )
          {
            unsigned long current_size = getFileSize(stream->filename);
            if (
                (current_size != stream->fileSize) ||
                (stream->autoRefreshForce)
               )
             {
              refreshVirtualStream(stream,stream->associatedModelList);
              stream->lastRefresh = timeAbsMilliseconds;
              stream->autoRefreshForce=0;
             }
          }
    }

   unsigned int FrameIDToReturn = 0;
   unsigned int FrameIDLast = 0;
   unsigned int FrameIDNext = 0;


   /*!OK , Two major cases here..! The one is a simple Next frame getter , the second is a more complicated interpolated frame getter..! */
   if ( (stream->object[ObjID].MAX_numberOfFrames == 0 ) )
   {
       fprintf(stderr,"Returning Null position for ObjID %u\n",ObjID);
       fillPosWithNull(/*stream,ObjID,*/pos,scaleX,scaleY,scaleZ);
       return 1;
   } else
   if  ( (stream->ignoreTime) || (stream->object[ObjID].MAX_numberOfFrames == 1 ) || ((stream->alwaysShowLastFrame)) )
   {
    //We might want to ignore time and just return frame after frame on each call!
    //Also if we only got one frame for the object there is no point in trying to interpolate time etc.. so just handle things here..
    if ( stream->object[ObjID].lastFrame +1 >= stream->object[ObjID].MAX_numberOfFrames ) { stream->object[ObjID].lastFrame  = 0; }
    FrameIDToReturn = stream->object[ObjID].lastFrame;
    ++stream->object[ObjID].lastFrame;


    //OK new rules.. ------------------------------------------------------
    // If we ignore time we suppose that the time given is the frame number so we use this ..!
    FrameIDToReturn = timeAbsMilliseconds;


    if (stream->alwaysShowLastFrame)
    {
      if (stream->object[ObjID].numberOfFrames>0)
      {
       FrameIDToReturn = stream->object[ObjID].numberOfFrames-1;
       //fprintf(stderr,"Always showing last frame ( %u ) , F7 to change\n",FrameIDToReturn);
      }
    }


    //fprintf(stderr,"Simple Getter ObjID %u Frame %u\n",ObjID,FrameIDToReturn);
    fillPosWithFrame(
                      stream,
                      ObjID,
                      FrameIDToReturn,
                      pos,
                      joints,
                      scaleX,
                      scaleY,
                      scaleZ
                    );
    //fprintf(stderr,"fillPosWithFrame %u => ( %0.2f %0.2f %0.2f , %0.2f %0.2f %0.2f)\n",FrameIDToReturn,pos[0],pos[1],pos[2],pos[3],pos[4],pos[5]);

    FrameIDLast = FrameIDToReturn;
    FrameIDNext = FrameIDToReturn+1;

    if ( FrameIDNext >= stream->object[ObjID].numberOfFrames )
     { //We 've reached the end of the stream so the last frame should truncate 0
       stream->object[ObjID].lastFrame=0;
     }

     return 1;

   } /*!END OF SIMPLE FRAME GETTER*/
   else
   { /*!START OF INTERPOLATED FRAME GETTER*/

    //fprintf(stderr,"Interpolated Getter ObjID %u Frame %u\n",ObjID,FrameIDToReturn);
     //fprintf(stderr,"interpolated position for ObjID %u\n",ObjID);
     //This is the case when we respect time , we will pick two frames and interpolate between them
     if ( timeAbsMilliseconds > stream->object[ObjID].MAX_timeOfFrames )
     {
       //This means we have passed the last frame.. so lets find out where we really are..
       if (stream->object[ObjID].MAX_timeOfFrames == 0 ) {
                                                           //If max time of frames is 0 then our time is also zero ( since it never goes over max )
                                                           //fprintf(stderr,"timeAbsMilliseconds can not be something more than zero");
                                                           timeAbsMilliseconds=0;
                                                         } else
                                                         { timeAbsMilliseconds = timeAbsMilliseconds % stream->object[ObjID].MAX_timeOfFrames; }
       //timeAbsMilliseconds should contain a valid value now somewhere from 0->MAX_timeOfFrames
     }

     #if PRINT_DEBUGGING_INFO
     fprintf(stderr,"Object %u has %u frames , lets search where we are \n",ObjID,stream->object[ObjID].numberOfFrames);
     #endif

     //We scan all the frames to find out the "last one" and the "next one"
     unsigned int i =0;
     for ( i=0; i <stream->object[ObjID].MAX_numberOfFrames-1; i++ )
      {
       if (( stream->object[ObjID].frame[i].time <= timeAbsMilliseconds )
                 &&
           ( timeAbsMilliseconds <= stream->object[ObjID].frame[i+1].time )  )
            {
               //This is the "next" frame!
               FrameIDLast = i;
               FrameIDNext = i+1;
               //This should be handled by raw response to zero elemetn :P
              break;
            }
      }

    //We now have our Last and Next frame , all that remains is extracting the
    //interpolated time between them..!
    return fillPosWithInterpolatedFrame(
                                         stream,
                                         ObjID,
                                         pos,
                                         joints,
                                         scaleX,
                                         scaleY,
                                         scaleZ,
                                         FrameIDLast,
                                         FrameIDNext,
                                         timeAbsMilliseconds
                                        );

   } /*!END OF INTERPOLATED FRAME GETTER*/

    return 0;
}



int calculateVirtualStreamPosAfterTime(
                                        struct VirtualStream * stream,
                                        ObjectIDHandler ObjID,
                                        unsigned int timeAfterMilliseconds,
                                        float * pos,
                                        float * joints,
                                        float * scaleX,
                                        float * scaleY,
                                        float * scaleZ
                                      )
{
   stream->object[ObjID].lastCalculationTime+=timeAfterMilliseconds;
   return calculateVirtualStreamPos(
                                     stream,
                                     ObjID,
                                     stream->object[ObjID].lastCalculationTime,
                                     pos,
                                     joints,
                                     scaleX,
                                     scaleY,
                                     scaleZ
                                    );
}


int getVirtualStreamLastPosF(
                              struct VirtualStream * stream,
                              ObjectIDHandler ObjID,
                              float * pos,
                              float * joints,
                              float * scaleX,
                              float * scaleY,
                              float * scaleZ
                            )
{
    return fillPosWithLastFrame(
                                 stream,
                                 ObjID,
                                 pos,
                                 joints,
                                 scaleX,
                                 scaleY,
                                 scaleZ
                                );
}




