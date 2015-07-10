#ifndef TRAJECTORYCALCULATOR_H_INCLUDED
#define TRAJECTORYCALCULATOR_H_INCLUDED

#include "TrajectoryParser.h"


int movePositionOfObjectTrajectorySt(struct VirtualStream * stream,unsigned int ObjID,unsigned int FrameIDToReturn,float relX,float relY,float relZ);
int movePositionOfObjectTrajectory(struct VirtualStream * stream,unsigned int ObjID,unsigned int FrameIDToReturn,float * relX,float * relY,float * relZ);
int rotatePositionOfObjectTrajectory(struct VirtualStream * stream,unsigned int ObjID,unsigned int FrameIDToReturn,float *x,float *y,float *z,float *angleDegrees);

int smoothTrajectoriesOfObject(struct VirtualStream * stream,unsigned int ObjID);
int smoothTrajectories(struct VirtualStream * stream);
float calculateDistanceTra(float from_x,float from_y,float from_z,float to_x,float to_y,float to_z);
void euler2QuaternionsInternal(double * quaternions,double * euler,int quaternionConvention);
int affixSatteliteToPlanetFromFrameForLength(struct VirtualStream * stream,unsigned int satteliteObj,unsigned int planetObj , unsigned int frameNumber , unsigned int duration);
int objectsCollide(struct VirtualStream * newstream,unsigned int atTime,unsigned int objIDA,unsigned int objIDB);

int flipRotationAxisD(double * rotX, double * rotY , double * rotZ , int where2SendX , int where2SendY , int where2SendZ);
int flipRotationAxis(float * rotX, float * rotY , float * rotZ , int where2SendX , int where2SendY , int where2SendZ);
int unflipRotationAxis(float * rotX, float * rotY , float * rotZ , int where2SendX , int where2SendY , int where2SendZ);




int convertQuaternionsToEulerAngles(struct VirtualStream * stream,double * euler,double *quaternion);





int fillPosWithNull(float * pos,float * scaleX ,float * scaleY,float * scaleZ);
int fillPosWithLastFrame(struct VirtualStream * stream,ObjectIDHandler ObjID,float * pos,float * scaleX,float * scaleY,float * scaleZ );
/*
int fillPosWithLastFrameD(struct VirtualStream * stream,ObjectIDHandler ObjID,double * pos,double * scale )
*/


int fillPosWithFrame(struct VirtualStream * stream,ObjectIDHandler ObjID,unsigned int FrameIDToReturn,float * pos,float * scaleX,float * scaleY,float * scaleZ);
int fillPosWithInterpolatedFrame(struct VirtualStream * stream,ObjectIDHandler ObjID,float * pos,float * scaleX,float * scaleY,float * scaleZ,
                                 unsigned int PrevFrame,unsigned int NextFrame , unsigned int time );
int calculateVirtualStreamPos(struct VirtualStream * stream,ObjectIDHandler ObjID,unsigned int timeAbsMilliseconds,float * pos,float * scaleX,float * scaleY,float * scaleZ);

int calculateVirtualStreamPosAfterTime(struct VirtualStream * stream,ObjectIDHandler ObjID,unsigned int timeAfterMilliseconds,float * pos,float * scaleX,float * scaleY, float * scaleZ);
int getVirtualStreamLastPosF(struct VirtualStream * stream,ObjectIDHandler ObjID,float * pos,float * scaleX,float * scaleY,float * scaleZ);


#endif // TRAJECTORYCALCULATOR_H_INCLUDED
