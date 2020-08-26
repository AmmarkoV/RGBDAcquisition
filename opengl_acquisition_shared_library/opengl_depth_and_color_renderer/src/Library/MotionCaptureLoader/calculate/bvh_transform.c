#include <stdio.h>
#include <math.h>
#include "bvh_transform.h"

#include "../../../../../../tools/AmMatrix/matrix4x4Tools.h"
 

//Also find Center of Joint
//We can skip the matrix multiplication by just grabbing the last column..
#define FIND_FAST_CENTER 1

//Also find Center of Joint
//We can skip the matrix multiplication by just grabbing the last column..
#define FAST_OFFSET_TRANSLATION 1


 

float max(float a,float b)
{
  if (a>b) {return a;}
  return b;
}

float min(float a,float b)
{
  if (a<b) {return a;}
  return b;
}


int bvh_populateTorso3DFromTransform(
                                      struct BVH_MotionCapture * mc ,
                                      struct BVH_Transform * bvhTransform
                                    )
{
 bvhTransform->torso.exists=0;
 bvhTransform->torso.rectangle2D.calculated=0;


 unsigned int jID=0;
 //Second test occlusions with torso..!
       //-------------------------------------------------------------
       int found=0;
       if ( bvh_getJointIDFromJointName(mc,"lshoulder",&jID) ) { found=1; } else
       if ( bvh_getJointIDFromJointName(mc,"lShldr",&jID) )    { found=1; }

       if (found)
       {
           bvhTransform->torso.point1Exists=1;
           bvhTransform->torso.rectangle3D.x1=bvhTransform->joint[jID].pos3D[0];
           bvhTransform->torso.rectangle3D.y1=bvhTransform->joint[jID].pos3D[1];
           bvhTransform->torso.rectangle3D.z1=bvhTransform->joint[jID].pos3D[2];
           bvhTransform->torso.jID[0]=jID;
       }
       //---
       found=0;
       if ( bvh_getJointIDFromJointName(mc,"rshoulder",&jID) ) { found=1; } else
       if ( bvh_getJointIDFromJointName(mc,"rShldr",&jID) )    { found=1; }

       if (found)
       {
           bvhTransform->torso.point2Exists=1;
           bvhTransform->torso.rectangle3D.x2=bvhTransform->joint[jID].pos3D[0];
           bvhTransform->torso.rectangle3D.y2=bvhTransform->joint[jID].pos3D[1];
           bvhTransform->torso.rectangle3D.z2=bvhTransform->joint[jID].pos3D[2];
           bvhTransform->torso.jID[1]=jID;
       }
       //---
       found=0;
       if ( bvh_getJointIDFromJointName(mc,"rhip",&jID) )      { found=1; } else
       if ( bvh_getJointIDFromJointName(mc,"rThigh",&jID) )    { found=1; }

       if (found)
       {
           bvhTransform->torso.point3Exists=1;
           bvhTransform->torso.rectangle3D.x3=bvhTransform->joint[jID].pos3D[0];
           bvhTransform->torso.rectangle3D.y3=bvhTransform->joint[jID].pos3D[1];
           bvhTransform->torso.rectangle3D.z3=bvhTransform->joint[jID].pos3D[2];
           bvhTransform->torso.jID[2]=jID;
       }
       //---
       found=0;
       if ( bvh_getJointIDFromJointName(mc,"lhip",&jID) )      { found=1; } else
       if ( bvh_getJointIDFromJointName(mc,"lThigh",&jID) )    { found=1; }

       if (found)
       {
           bvhTransform->torso.point4Exists=1;
           bvhTransform->torso.rectangle3D.x4=bvhTransform->joint[jID].pos3D[0];
           bvhTransform->torso.rectangle3D.y4=bvhTransform->joint[jID].pos3D[1];
           bvhTransform->torso.rectangle3D.z4=bvhTransform->joint[jID].pos3D[2];
           bvhTransform->torso.jID[3]=jID;
       }
       //---

       if (
            (bvhTransform->torso.point1Exists) &&
            (bvhTransform->torso.point2Exists) &&
            (bvhTransform->torso.point3Exists) &&
            (bvhTransform->torso.point4Exists)
          )
         {
            bvhTransform->torso.exists=1;
            bvhTransform->torso.averageDepth = ( bvhTransform->torso.rectangle3D.z1 +
                                                 bvhTransform->torso.rectangle3D.z2 +
                                                 bvhTransform->torso.rectangle3D.z3 +
                                                 bvhTransform->torso.rectangle3D.z4 ) / 4;
            return 1;
         }
       //-------------------------------------------------------------

  //fprintf(stderr,"%u %u %u %u\n",bvhTransform->torso.point1Exists,bvhTransform->torso.point2Exists,bvhTransform->torso.point3Exists,bvhTransform->torso.point4Exists);
  return 0;
}




int bvh_populateRectangle2DFromProjections(
                                           struct BVH_MotionCapture * mc ,
                                           struct BVH_Transform * bvhTransform,
                                           struct rectangleArea * area
                                          )
{
  if (area->exists)
  {
    unsigned int jID=0;
    unsigned int existing2DPoints=0;

    //-----------------------------------------------------------
    jID=area->jID[0];
    if (bvhTransform->joint[jID].pos2DCalculated)
    {
        area->rectangle2D.x1=bvhTransform->joint[jID].pos2D[0];
        area->rectangle2D.y1=bvhTransform->joint[jID].pos2D[1];
        ++existing2DPoints;
    }
    //-----------------------------------------------------------
    jID=area->jID[1];
    if (bvhTransform->joint[jID].pos2DCalculated)
    {
        area->rectangle2D.x2=bvhTransform->joint[jID].pos2D[0];
        area->rectangle2D.y2=bvhTransform->joint[jID].pos2D[1];
        ++existing2DPoints;
    }

    //-----------------------------------------------------------
    jID=area->jID[2];
    if (bvhTransform->joint[jID].pos2DCalculated)
    {
        area->rectangle2D.x3=bvhTransform->joint[jID].pos2D[0];
        area->rectangle2D.y3=bvhTransform->joint[jID].pos2D[1];
        ++existing2DPoints;
    }

    //-----------------------------------------------------------
    jID=area->jID[3];
    if (bvhTransform->joint[jID].pos2DCalculated)
    {
        area->rectangle2D.x4=bvhTransform->joint[jID].pos2D[0];
        area->rectangle2D.y4=bvhTransform->joint[jID].pos2D[1];
        ++existing2DPoints;
    }

    if ( existing2DPoints == 4 )
    {
      area->rectangle2D.calculated=1;

      float minimumX=min(area->rectangle2D.x1,min(area->rectangle2D.x2,min(area->rectangle2D.x3,area->rectangle2D.x4)));
      float minimumY=min(area->rectangle2D.y1,min(area->rectangle2D.y2,min(area->rectangle2D.y3,area->rectangle2D.y4)));
      float maximumX=max(area->rectangle2D.x1,max(area->rectangle2D.x2,max(area->rectangle2D.x3,area->rectangle2D.x4)));
      float maximumY=max(area->rectangle2D.y1,max(area->rectangle2D.y2,max(area->rectangle2D.y3,area->rectangle2D.y4)));
      area->rectangle2D.x=minimumX;
      area->rectangle2D.y=minimumY;
      area->rectangle2D.width=maximumX-minimumX;
      area->rectangle2D.height=maximumY-minimumY;
      return 1;
    } else
    {
        fprintf(stderr,"Only found %u/4 of the joints needed to get a rectangle \n",existing2DPoints);
    }
  } else
  {
    //Less spam..
   //fprintf(stderr,"bvh_populateRectangle2DFromProjections: Area does not exist..\n");
  }

 return 0;
}

 


unsigned char bvh_shouldJointBeTransformedGivenOurOptimizations(const struct BVH_Transform * bvhTransform,const BVHJointID jID)
{ 
  //Normaly we should check if the bvhTransform structure is null but given that this call is called millions of times only if you have a transform
  //we skip the check
  //if (bvhTransform==0) { return 0; }
 
  if (bvhTransform->useOptimizations) 
  {
     //If we are using optimizations and this joint is not skipped then transform this joint
     //Normally we should check for index errors but in an effort to speed up the function to the maximum extent the check is skipped
     //if (jID<MAX_BVH_JOINT_HIERARCHY_SIZE)  
     //     { 
            return  (!bvhTransform->skipCalculationsForJoint[jID]); 
     //     }
  } else
  {
     //If we are not using optimizations then transform every joint 
     return 1;
  } 
}



void bvh_printNotSkippedJoints(struct BVH_MotionCapture * bvhMotion ,struct BVH_Transform * bvhTransform)
{ 
   for (BVHJointID jID=0; jID<bvhMotion->jointHierarchySize; jID++)
   {
     if (!bvhTransform->skipCalculationsForJoint[jID])
     {
         fprintf(stderr,"Joint %u ( %s ) is selected \n" ,jID , bvhMotion->jointHierarchy[jID].jointName);
     }
   }
}



void bvh_HashUsefulJoints(struct BVH_MotionCapture * bvhMotion,struct BVH_Transform * bvhTransform)
{
   #if USE_TRANSFORM_HASHING
   /*
   //This is a little stupid.. if we are not using optimizations then we go through
   //all elements..
   if (!bvhTransform->useOptimizations)
   {    
      unsigned int hashedElementsCounter=0;
      for (BVHJointID jID=0; jID<bvhMotion->jointHierarchySize; jID++)
       {
         bvhTransform->listOfJointIDsToTransform[hashedElementsCounter]=jID;
         ++hashedElementsCounter;
       }
       
     bvhTransform->lengthOfListOfJointIDsToTransform=hashedElementsCounter; //Start from the .. start.. 
     bvhTransform->jointIDTransformHashPopulated=1; //(hashedElementsCounter>0); //Mark the hash as populated
     return ;
   }
   */
    
   //Since we do several million accesses an additional optimization is to keep a list of interesting joints
   //and only access them
   //-----------------------------------------------------------------------------------------------
   unsigned int hashedElementsCounter=0;
   for (BVHJointID jID=0; jID<bvhMotion->jointHierarchySize; jID++)
   {
       if ( (!bvhTransform->useOptimizations) || (!bvhTransform->skipCalculationsForJoint[jID]) )
       {
         bvhTransform->listOfJointIDsToTransform[hashedElementsCounter]=jID;
         ++hashedElementsCounter;
       }
   }
   bvhTransform->lengthOfListOfJointIDsToTransform=hashedElementsCounter; //Start from the .. start.. 
   bvhTransform->jointIDTransformHashPopulated=1; //(hashedElementsCounter>0); //Mark the hash as populated
   //-----------------------------------------------------------------------------------------------
   #endif
   return;
}



int bvh_markAllJointsAsUsefullInTransform(
                                          struct BVH_MotionCapture * bvhMotion ,
                                          struct BVH_Transform * bvhTransform
                                         )
{
  if (bvhMotion==0)    { return 0; }
  if (bvhTransform==0) { return 0; }
  bvhTransform->useOptimizations=1;

   for (BVHJointID jID=0; jID<bvhMotion->jointHierarchySize; jID++)
   {
     bvhTransform->skipCalculationsForJoint[jID]=0;
   }
   
  #if USE_TRANSFORM_HASHING
   //Marking all joints as useless is the same as setting the length of the joint 
   //list to zero
   bvh_HashUsefulJoints(bvhMotion,bvhTransform);
  #endif
 
  return 1;
}


int bvh_markAllJointsAsUselessInTransform(
                                          struct BVH_MotionCapture * bvhMotion ,
                                          struct BVH_Transform * bvhTransform
                                         )
{
  if (bvhMotion==0)    { return 0; }
  if (bvhTransform==0) { return 0; }
  bvhTransform->useOptimizations=1;

   for (BVHJointID jID=0; jID<bvhMotion->jointHierarchySize; jID++)
   {
     bvhTransform->skipCalculationsForJoint[jID]=1;
   }
   
  #if USE_TRANSFORM_HASHING
   //Marking all joints as useless is the same as setting the length of the joint 
   //list to zero
   bvhTransform->lengthOfListOfJointIDsToTransform=0; //No joints are interesting
  #endif
 
  return 1;
}


int bvh_markJointAndParentsAsUsefulInTransform(
                                                struct BVH_MotionCapture * bvhMotion ,
                                                struct BVH_Transform * bvhTransform,
                                                BVHJointID jID
                                              )
{
  if (bvhMotion==0)    { return 0; }
  if (bvhTransform==0) { return 0; }
  if (jID>=bvhMotion->jointHierarchySize) { return 0; }
  bvhTransform->useOptimizations=1;
 
  //We want to make sure all parent joints until root ( jID->0 ) are set to not skip calculations..
  while (jID!=0)
      {
           bvhTransform->skipCalculationsForJoint[jID]=0;
           jID = bvhMotion->jointHierarchy[jID].parentJoint;
      }
 
  bvhTransform->skipCalculationsForJoint[0]=0;

  #if USE_TRANSFORM_HASHING
  //As an extra speed up we hash the interesting joints
   bvh_HashUsefulJoints(bvhMotion,bvhTransform);
  #endif
 
  return 1;
}


int bvh_markJointAndParentsAsUselessInTransform(
                                                struct BVH_MotionCapture * bvhMotion ,
                                                struct BVH_Transform * bvhTransform,
                                                BVHJointID jID
                                              )
{
  if (bvhMotion==0)    { return 0; }
  if (bvhTransform==0) { return 0; }
  if (jID>=bvhMotion->jointHierarchySize) { return 0; }
  bvhTransform->useOptimizations=1;
   

  while (jID!=0)
      {
           bvhTransform->skipCalculationsForJoint[jID]=1;
           jID = bvhMotion->jointHierarchy[jID].parentJoint;
      }

  bvhTransform->skipCalculationsForJoint[0]=1;

  #if USE_TRANSFORM_HASHING
   //As an extra speed up we hash the interesting joints
   bvh_HashUsefulJoints(bvhMotion,bvhTransform);
  #endif
  
  return 1;
}


int bvh_markJointAsUsefulAndParentsAsUselessInTransform(
                                                        struct BVH_MotionCapture * bvhMotion ,
                                                        struct BVH_Transform * bvhTransform,
                                                        BVHJointID jID
                                                       )
{
  if (bvhMotion==0)    { return 0; }
  if (bvhTransform==0) { return 0; }
  if (jID>=bvhMotion->jointHierarchySize) { return 0; }

  bvhTransform->useOptimizations=1;
  bvh_markJointAndParentsAsUselessInTransform(bvhMotion,bvhTransform,jID);
  bvhTransform->skipCalculationsForJoint[jID]=0;

  #if USE_TRANSFORM_HASHING
   //As an extra speed up we hash the interesting joints
   bvh_HashUsefulJoints(bvhMotion,bvhTransform);
  #endif
  
  return 1;
}



/**
 * @brief This is a MocapNET orientation.
 */
enum MOTIONBUFFER_DATA_FIELDS
{
 MOTIONBUFFER_DATA_FIELDS_POSX=0,
 MOTIONBUFFER_DATA_FIELDS_POSY,
 MOTIONBUFFER_DATA_FIELDS_POSZ,
 MOTIONBUFFER_DATA_FIELDS_ROTX,
 MOTIONBUFFER_DATA_FIELDS_ROTY,
 MOTIONBUFFER_DATA_FIELDS_ROTZ,
 //-----------------------------
 MOTIONBUFFER_DATA_FIELDS_NUMBER
};


int bvh_loadTransformForMotionBuffer(
                                     struct BVH_MotionCapture * bvhMotion,
                                     float * motionBuffer,
                                     struct BVH_Transform * bvhTransform,
                                     unsigned int populateTorso
                                   )
{
  //Only do transforms on allocated context 
  if ( (bvhMotion!=0) && (motionBuffer!=0) && (bvhTransform!=0))
  {
    
  //First of all we need to clean the BVH_Transform structure
  bvhTransform->jointsOccludedIn2DProjection=0;


  //data is the buffer where we will retrieve the values
  float data[MOTIONBUFFER_DATA_FIELDS_NUMBER]={0};
  //----------------------------------------------------

  //First of all we need to populate all local dynamic transformation of our chain
  //These only have to do with our Motion Buffer and don't involve any chain transformations
  //----------------------------------------------------------------------------------------
  #if USE_TRANSFORM_HASHING
   for (unsigned int hashID=0; hashID<bvhTransform->lengthOfListOfJointIDsToTransform; hashID++)
   {
    //USING_HASH   
    unsigned int jID=bvhTransform->listOfJointIDsToTransform[hashID];
  #else 
   for (unsigned int jID=0; jID<bvhMotion->jointHierarchySize; jID++)
   {
  #endif
    if (bvh_shouldJointBeTransformedGivenOurOptimizations(bvhTransform,jID))
    {
      //To Setup the dynamic transformation we must first get values from our bvhMotion structure
      if (bhv_populatePosXYZRotXYZFromMotionBuffer(bvhMotion,jID,motionBuffer,data,sizeof(data)))
      {
       create4x4FTranslationMatrix(
                                    &bvhTransform->joint[jID].dynamicTranslation,
                                    data[MOTIONBUFFER_DATA_FIELDS_POSX],
                                    data[MOTIONBUFFER_DATA_FIELDS_POSY],
                                    data[MOTIONBUFFER_DATA_FIELDS_POSZ]
                                  );


       if ( (bvhMotion->jointHierarchy[jID].channelRotationOrder!=0)  ) 
       {
        create4x4FMatrixFromEulerAnglesWithRotationOrder(
                                                        &bvhTransform->joint[jID].dynamicRotation,
                                                        -1*data[MOTIONBUFFER_DATA_FIELDS_ROTX],
                                                        -1*data[MOTIONBUFFER_DATA_FIELDS_ROTY],
                                                        -1*data[MOTIONBUFFER_DATA_FIELDS_ROTZ],
                                                        (unsigned int) bvhMotion->jointHierarchy[jID].channelRotationOrder
                                                       ); 
       } else
       {
         //No rotation order will get you an Identity rotation   
         create4x4FIdentityMatrix(&bvhTransform->joint[jID].dynamicRotation);
         
         //This is normal in an end-site but we should 
         //display a debug message in case this doesn't happen on an end site
         if (!bvhMotion->jointHierarchy[jID].isEndSite)
              {
                fprintf(stderr,"No channel rotation order for joint jID=%u jointName=%s, using identity matrix\n",jID,bvhMotion->jointHierarchy[jID].jointName);
              }
       } 
       
       
     } else
     {
      fprintf(stderr,"Error extracting dynamic transformation for jID=%u and a motionBuffer\n",jID);
      create4x4FIdentityMatrix(&bvhTransform->joint[jID].dynamicTranslation);
      create4x4FIdentityMatrix(&bvhTransform->joint[jID].dynamicRotation);
     }
    }
  }




  //We will now apply all dynamic transformations across the BVH chains
  //-----------------------------------------------------------------------
  #if USE_TRANSFORM_HASHING
   for (unsigned int hashID=0; hashID<bvhTransform->lengthOfListOfJointIDsToTransform; hashID++)
   {  
     //USING_HASH
     unsigned int jID=bvhTransform->listOfJointIDsToTransform[hashID];
  #else 
   for (unsigned int jID=0; jID<bvhMotion->jointHierarchySize; jID++)
   {
  #endif
    if (bvh_shouldJointBeTransformedGivenOurOptimizations(bvhTransform,jID))
    {
     //This will get populated either way..
     //create4x4FIdentityMatrix(bvhTransform->joint[jID].localToWorldTransformation);

     if (bhv_jointHasParent(bvhMotion,jID))
      {
        //If joint is not Root joint
        unsigned int parentID = bvhMotion->jointHierarchy[jID].parentJoint;


        if (!bvhTransform->joint[parentID].isChainTrasformationComputed)
        {
         //This is needed because we access the chain transform of our parent so at some point this will get used..
         bvhTransform->joint[parentID].isChainTrasformationComputed=1;
         create4x4FIdentityMatrix(&bvhTransform->joint[parentID].chainTransformation);
        }


        multiplyTwo4x4FMatricesS(
                                //Output AxB
                                &bvhTransform->joint[jID].localToWorldTransformation ,
                                //Parent Output A
                                &bvhTransform->joint[parentID].chainTransformation,
                                //This Transform B
                                &bvhMotion->jointHierarchy[jID].staticTransformation
                              );
      } else
      if ( bvhMotion->jointHierarchy[jID].isRoot)
      {
       //If we are the root node there is no parent..
       //If there is no parent we will only set our position and copy to the final transform
        #if FAST_OFFSET_TRANSLATION
         //Skip the matrix multiplication..
         create4x4FTranslationMatrix(
                                     &bvhTransform->joint[jID].localToWorldTransformation,
                                     bvhMotion->jointHierarchy[jID].staticTransformation.m[3]  + bvhTransform->joint[jID].dynamicTranslation.m[3],
                                     bvhMotion->jointHierarchy[jID].staticTransformation.m[7]  + bvhTransform->joint[jID].dynamicTranslation.m[7],
                                     bvhMotion->jointHierarchy[jID].staticTransformation.m[11] + bvhTransform->joint[jID].dynamicTranslation.m[11]
                                    );
        #else
         //We can do the matrix multipliaction to calculate the offset..
         multiplyTwo4x4FMatrices(
                                //Output AxB
                                bvhTransform->joint[jID].localToWorldTransformation ,
                                //A
                                bvhMotion->jointHierarchy[jID].staticTransformation,
                                //B
                                bvhTransform->joint[jID].dynamicTranslation
                              );
        #endif // FAST_OFFSET_TRANSLATION
      } else
      {
        //Weird case where joint is not root and doesnt have parents(?)
        create4x4FIdentityMatrix(&bvhTransform->joint[jID].localToWorldTransformation);
      }

    bvhTransform->joint[jID].isChainTrasformationComputed=1;
    multiplyTwo4x4FMatricesS(
                           //Output AxB
                           &bvhTransform->joint[jID].chainTransformation ,
                           //A
                           &bvhTransform->joint[jID].localToWorldTransformation,
                           //B
                           &bvhTransform->joint[jID].dynamicRotation
                          );

  #if FIND_FAST_CENTER
   bvhTransform->joint[jID].pos3D[0]=bvhTransform->joint[jID].localToWorldTransformation.m[3];
   bvhTransform->joint[jID].pos3D[1]=bvhTransform->joint[jID].localToWorldTransformation.m[7];
   bvhTransform->joint[jID].pos3D[2]=bvhTransform->joint[jID].localToWorldTransformation.m[11];
   bvhTransform->joint[jID].pos3D[3]=bvhTransform->joint[jID].localToWorldTransformation.m[15]; 
   normalize3DPointFVector(bvhTransform->joint[jID].pos3D);
  #else
   float centerPoint[4]={0.0,0.0,0.0,1.0};
   transform3DPointFVectorUsing4x4FMatrix(
                                          bvhTransform->joint[jID].pos3D,
                                          bvhTransform->joint[jID].localToWorldTransformation,
                                          centerPoint
                                         );
   normalize3DPointFVector(bvhTransform->joint[jID].pos3D);
  #endif // FIND_FAST_CENTER

    }
  }


  bvhTransform->centerPosition[0]=bvhTransform->joint[bvhMotion->rootJointID].pos3D[0];
  bvhTransform->centerPosition[1]=bvhTransform->joint[bvhMotion->rootJointID].pos3D[1];
  bvhTransform->centerPosition[2]=bvhTransform->joint[bvhMotion->rootJointID].pos3D[2];


  if (!populateTorso)
  {
    //Fast path out of here
    return 1;      
  } else
  {
   if (!bvh_populateTorso3DFromTransform(bvhMotion,bvhTransform))
     {
     //fprintf(stderr,"bvh_loadTransformForMotionBuffer: Could not populate torso information from 3D transform\n");
     }
    return 1;   
  }


 }
   
 return 0;
}



int bvh_loadTransformForFrame(
                               struct BVH_MotionCapture * bvhMotion,
                               BVHFrameID fID ,
                               struct BVH_Transform * bvhTransform,
                               unsigned int populateTorso
                             )
{
  int result = 0;
  struct MotionBuffer * frameMotionBuffer  = mallocNewMotionBuffer(bvhMotion);

  if (frameMotionBuffer!=0)
  {
    if  ( bvh_copyMotionFrameToMotionBuffer(bvhMotion,frameMotionBuffer,fID)  )
    {
     result = bvh_loadTransformForMotionBuffer(
                                               bvhMotion ,
                                               frameMotionBuffer->motion,
                                               bvhTransform,
                                               populateTorso
                                              );
    }
    freeMotionBuffer(frameMotionBuffer);
  }

  return result;
}





int bvh_removeTranslationFromTransform(
                                       struct BVH_MotionCapture * bvhMotion ,
                                       struct BVH_Transform * bvhTransform
                                      )
{
  fprintf(stderr,"bvh_removeTranslationFromTransform not correctly implemented");
  BVHJointID rootJID=0;

  if ( bvh_getRootJointID(bvhMotion,&rootJID) )
  {
   float rX = bvhTransform->joint[rootJID].pos3D[0];
   float rY = bvhTransform->joint[rootJID].pos3D[1];
   float rZ = bvhTransform->joint[rootJID].pos3D[2];
      
   float r2DX = bvhTransform->joint[rootJID].pos2D[0];
   float r2DY = bvhTransform->joint[rootJID].pos2D[1];
     
   for (BVHJointID jID=0; jID<bvhMotion->jointHierarchySize; jID++)
    {
     bvhTransform->joint[jID].pos3D[0]=bvhTransform->joint[jID].pos3D[0]-rX;
     bvhTransform->joint[jID].pos3D[1]=bvhTransform->joint[jID].pos3D[1]-rY;
     bvhTransform->joint[jID].pos3D[2]=bvhTransform->joint[jID].pos3D[2]-rZ;

     bvhTransform->joint[jID].pos2D[0]=bvhTransform->joint[jID].pos2D[0]-r2DX;
     bvhTransform->joint[jID].pos2D[1]=bvhTransform->joint[jID].pos2D[1]-r2DY;
    }
  }

  return 0;
}


