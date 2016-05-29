#ifndef SKELETON_H_INCLUDED
#define SKELETON_H_INCLUDED

#include <math.h>

#ifdef __cplusplus
extern "C"
{
#endif




static float defaultJoints[] = { 70.37139892578125, -488.10699462890625, 1605.5, -29.082809448242188, -314.5812072753906, 1605.500732421875, -29.102794647216797, -85.98794555664062, 1566.6517333984375, -214.7298126220703, -312.88775634765625, 1617.931640625, 73.657470703125, -311.418701171875, 1617.529541015625, -252.35317993164062, -527.5241088867188, 1616.2698974609375, 64.68223571777344, -578.2517700195312, 1656.204345703125, -245.8188018798828, -441.24560546875, 1435.18798828125, 128.5203857421875, -492.1353759765625, 1511.617431640625, -120.2656021118164, 128.38536071777344, 1526.885498046875, 62.020042419433594, 156.82525634765625, 1528.719970703125, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -29.122779846191406, 142.60531616210938, 1527.802734375 };




static const char * jointNames[] =
{
 "head",
 "neck" ,
 "torso",
 "right_shoulder",
 "left_shoulder",
 "right_elbow",
 "left_elbow",
 "right_hand",
 "left_hand",
 "right_hip",
 "left_hip",
 "right_knee",
 "left_knee",
 "right_foot",
 "left_foot" ,
 "hip" ,
 //=================
 "End of Joint Names"
};


static const char * tgbtNames[] =
{
 "head",
 "neck",
 "bodyCenter",
 "rightShoulder",
 "leftShoulder",
 "rightElbow",
 "leftElbow",
 "rightWrist",
 "leftWrist",
 "rightLegRoot",
 "leftLegRoot",
 "rightKnee",
 "leftKnee",
 "rightAnkle",
 "leftAnkle",
 "hip",
 //--------------------
 "End Of TGBT Names"
};

static const char * const humanSkeletonJointNames[] =
    {
       "HUMAN_SKELETON_HEAD",
       "HUMAN_SKELETON_NECK",
       "HUMAN_SKELETON_TORSO",
       "HUMAN_SKELETON_RIGHT_SHOULDER",
       "HUMAN_SKELETON_LEFT_SHOULDER",
       "HUMAN_SKELETON_RIGHT_ELBOW",
       "HUMAN_SKELETON_LEFT_ELBOW",
       "HUMAN_SKELETON_RIGHT_HAND",
       "HUMAN_SKELETON_LEFT_HAND",
       "HUMAN_SKELETON_RIGHT_HIP",
       "HUMAN_SKELETON_LEFT_HIP",
       "HUMAN_SKELETON_RIGHT_KNEE",
       "HUMAN_SKELETON_LEFT_KNEE",
       "HUMAN_SKELETON_RIGHT_FOOT",
       "HUMAN_SKELETON_LEFT_FOOT",
       "HUMAN_SKELETON_HIP"
    };

enum humanSkeletonJoints
{
   HUMAN_SKELETON_HEAD = 0,
   HUMAN_SKELETON_NECK,
   HUMAN_SKELETON_TORSO,
   HUMAN_SKELETON_RIGHT_SHOULDER,
   HUMAN_SKELETON_LEFT_SHOULDER,
   HUMAN_SKELETON_RIGHT_ELBOW,
   HUMAN_SKELETON_LEFT_ELBOW,
   HUMAN_SKELETON_RIGHT_HAND,
   HUMAN_SKELETON_LEFT_HAND,
   HUMAN_SKELETON_RIGHT_HIP,
   HUMAN_SKELETON_LEFT_HIP,
   HUMAN_SKELETON_RIGHT_KNEE,
   HUMAN_SKELETON_LEFT_KNEE,
   HUMAN_SKELETON_RIGHT_FOOT,
   HUMAN_SKELETON_LEFT_FOOT,
   HUMAN_SKELETON_HIP,
   //---------------------
   HUMAN_SKELETON_PARTS
};


static const int humanSkeletonJointsRelationMap[] =
{ // Parent                        Joint
  HUMAN_SKELETON_HEAD,           //HUMAN_SKELETON_HEAD
  HUMAN_SKELETON_HEAD,           //HUMAN_SKELETON_NECK
  HUMAN_SKELETON_NECK,           //HUMAN_SKELETON_TORSO
  HUMAN_SKELETON_NECK,           //HUMAN_SKELETON_RIGHT_SHOULDER
  HUMAN_SKELETON_NECK,           //HUMAN_SKELETON_LEFT_SHOULDER
  HUMAN_SKELETON_RIGHT_SHOULDER, //HUMAN_SKELETON_RIGHT_ELBOW
  HUMAN_SKELETON_LEFT_SHOULDER,  //HUMAN_SKELETON_LEFT_ELBOW
  HUMAN_SKELETON_RIGHT_ELBOW,    //HUMAN_SKELETON_RIGHT_HAND
  HUMAN_SKELETON_LEFT_ELBOW,     //HUMAN_SKELETON_LEFT_HAND
  HUMAN_SKELETON_HIP,            //HUMAN_SKELETON_RIGHT_HIP
  HUMAN_SKELETON_HIP,            //HUMAN_SKELETON_LEFT_HIP
  HUMAN_SKELETON_RIGHT_HIP,      //HUMAN_SKELETON_RIGHT_KNEE
  HUMAN_SKELETON_LEFT_HIP,       //HUMAN_SKELETON_LEFT_KNEE
  HUMAN_SKELETON_LEFT_KNEE,      //HUMAN_SKELETON_RIGHT_FOOT
  HUMAN_SKELETON_LEFT_KNEE,      //HUMAN_SKELETON_LEFT_FOOT
  HUMAN_SKELETON_TORSO,          //HUMAN_SKELETON_HIP
};


static const char * const humanSkeletonMirroredJointNames[] =
    {
      "HUMAN_SKELETON_MIRRORED_HEAD",
      "HUMAN_SKELETON_MIRRORED_NECK",
      "HUMAN_SKELETON_MIRRORED_TORSO",
      "HUMAN_SKELETON_MIRRORED_LEFT_SHOULDER",
      "HUMAN_SKELETON_MIRRORED_RIGHT_SHOULDER",
      "HUMAN_SKELETON_MIRRORED_LEFT_ELBOW",
      "HUMAN_SKELETON_MIRRORED_RIGHT_ELBOW",
      "HUMAN_SKELETON_MIRRORED_LEFT_HAND",
      "HUMAN_SKELETON_MIRRORED_RIGHT_HAND",
      "HUMAN_SKELETON_MIRRORED_LEFT_HIP",
      "HUMAN_SKELETON_MIRRORED_RIGHT_HIP",
      "HUMAN_SKELETON_MIRRORED_LEFT_KNEE",
      "HUMAN_SKELETON_MIRRORED_RIGHT_KNEE",
      "HUMAN_SKELETON_MIRRORED_LEFT_FOOT",
      "HUMAN_SKELETON_MIRRORED_RIGHT_FOOT",
      "HUMAN_SKELETON_MIRRORED_HIP"
    };

enum humanMirroredSkeletonJoints
{
   HUMAN_SKELETON_MIRRORED_HEAD = 0,
   HUMAN_SKELETON_MIRRORED_NECK,
   HUMAN_SKELETON_MIRRORED_TORSO,
   HUMAN_SKELETON_MIRRORED_LEFT_SHOULDER,
   HUMAN_SKELETON_MIRRORED_RIGHT_SHOULDER,
   HUMAN_SKELETON_MIRRORED_LEFT_ELBOW,
   HUMAN_SKELETON_MIRRORED_RIGHT_ELBOW,
   HUMAN_SKELETON_MIRRORED_LEFT_HAND,
   HUMAN_SKELETON_MIRRORED_RIGHT_HAND,
   HUMAN_SKELETON_MIRRORED_LEFT_HIP,
   HUMAN_SKELETON_MIRRORED_RIGHT_HIP,
   HUMAN_SKELETON_MIRRORED_LEFT_KNEE,
   HUMAN_SKELETON_MIRRORED_RIGHT_KNEE,
   HUMAN_SKELETON_MIRRORED_LEFT_FOOT,
   HUMAN_SKELETON_MIRRORED_RIGHT_FOOT,
   HUMAN_SKELETON_MIRRORED_HIP,
   //---------------------
   HUMAN_SKELETON_MIRRORED_PARTS
};


struct point2D
{
    float x,y;
};

struct point3D
{
    float x,y,z;
};




struct skeletonNAO
{
  unsigned int observationNumber , observationTotal;
  unsigned int userID;

  unsigned char isNew,isVisible,isOutOfScene,isLost;
  unsigned char statusCalibrating,statusStoppedTracking, statusTracking,statusFailed;

  struct point3D bbox[8];
  struct point3D bboxDimensions;
  struct point3D centerOfMass;
  struct point3D joint[HUMAN_SKELETON_PARTS];
  struct point2D joint2D[HUMAN_SKELETON_PARTS];
  float jointAccuracy[HUMAN_SKELETON_PARTS];
};




struct skeletonHuman
{
  unsigned int observationNumber , observationTotal;
  unsigned int userID;

  unsigned char isNew,isVisible,isOutOfScene,isLost;
  unsigned char statusCalibrating,statusStoppedTracking, statusTracking,statusFailed;

  struct point3D bbox[8];
  struct point3D bboxDimensions;
  struct point3D centerOfMass;
  struct point3D joint[HUMAN_SKELETON_PARTS];
  struct point2D joint2D[HUMAN_SKELETON_PARTS];
  float  jointAccuracy[HUMAN_SKELETON_PARTS];

  struct point3D relativeJointAngle[HUMAN_SKELETON_PARTS];
};


struct skeletonPointing
{
  struct point3D pointStart;
  struct point3D pointEnd;
  struct point3D pointingVector;
  unsigned char isLeftHand;
  unsigned char isRightHand;
};


static int skeletonEmptyJoint(struct skeletonHuman * sk , unsigned int j1)
{
   if (
        (sk->joint[j1].x==0) &&
        (sk->joint[j1].y==0) &&
        (sk->joint[j1].z==0)
       )
   {
     return 1;
   }
  return 0;
}


static int skeletonSameJoints(unsigned int j1 , unsigned int j2)
{
   if (humanSkeletonJointsRelationMap[j1]==j2)
   {
     return 1;
   }
  return 0;
}

static void updateSkeletonBoundingBox(struct skeletonHuman * sk)
{
  //Use joints to extract bbox
  float minX = sk->joint[HUMAN_SKELETON_HEAD].x; float maxX = sk->joint[HUMAN_SKELETON_HEAD].x;
  float minY = sk->joint[HUMAN_SKELETON_HEAD].y; float maxY = sk->joint[HUMAN_SKELETON_HEAD].y;
  float minZ = sk->joint[HUMAN_SKELETON_HEAD].z; float maxZ = sk->joint[HUMAN_SKELETON_HEAD].z;

  unsigned int i=0;
     for (i=0; i<HUMAN_SKELETON_PARTS; i++)
      {
       if (
            (sk->joint[i].x!=0.0) ||
            (sk->joint[i].y!=0.0) ||
            (sk->joint[i].z!=0.0)
           )
        {
         if (sk->joint[i].x>maxX) { maxX = sk->joint[i].x; } else
         if (sk->joint[i].x<minX) { minX = sk->joint[i].x; }

         if (sk->joint[i].y>maxY) { maxY = sk->joint[i].y; } else
         if (sk->joint[i].y<minY) { minY = sk->joint[i].y; }

         if (sk->joint[i].z>maxZ) { maxZ = sk->joint[i].z; } else
         if (sk->joint[i].z<minZ) { minZ = sk->joint[i].z; }
        }
      }

     sk->bbox[0].x = maxX; sk->bbox[0].y = maxY; sk->bbox[0].z = minZ;
     sk->bbox[1].x = maxX; sk->bbox[1].y = minY; sk->bbox[1].z = minZ;
     sk->bbox[2].x = minX; sk->bbox[2].y = minY; sk->bbox[2].z = minZ;
     sk->bbox[3].x = minX; sk->bbox[3].y = maxY; sk->bbox[3].z = minZ;
     sk->bbox[4].x = maxX; sk->bbox[4].y = maxY; sk->bbox[4].z = maxZ;
     sk->bbox[5].x = maxX; sk->bbox[5].y = minY; sk->bbox[5].z = maxZ;
     sk->bbox[6].x = minX; sk->bbox[6].y = minY; sk->bbox[6].z = maxZ;
     sk->bbox[7].x = minX; sk->bbox[7].y = maxY; sk->bbox[7].z = maxZ;

     sk->bboxDimensions.x = (float) sk->bbox[4].x-sk->bbox[2].x;
     sk->bboxDimensions.y = (float) sk->bbox[4].y-sk->bbox[2].y;
     sk->bboxDimensions.z = (float) sk->bbox[4].z-sk->bbox[2].z;
}







static double getAngleABC( double* srcPoint, double* dstPoint, double* defaultPoint )
{
 double *a = srcPoint, *b = dstPoint , *c = defaultPoint;
/*
                      * dstPoint
                    /
                /
            /
 srcPoint/
      *  -  -  -  -  -  *  defaultPoint

*/
    double ab[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
    double bc[3] = { c[0] - b[0], c[1] - b[1], c[2] - b[2]  };

    double abVec = sqrt(ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]);
    double bcVec = sqrt(bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]);

    double abNorm[3] = {ab[0] / abVec, ab[1] / abVec, ab[2] / abVec};
    double bcNorm[3] = {bc[0] / bcVec, bc[1] / bcVec, bc[2] / bcVec};

    double res = abNorm[0] * bcNorm[0] + abNorm[1] * bcNorm[1] + abNorm[2] * bcNorm[2];

    return acos(res)*180.0/ 3.141592653589793;
}



static double getAngleABCRelative( double* srcPoint, double* dstPoint, double* srcDefaultPoint , double* dstDefaultPoint  )
{
 double relativeSrc = 0;
 double relativeDst = *dstPoint - * srcPoint;
 double relativeDefaultDst = *dstDefaultPoint - * srcDefaultPoint;


 return getAngleABC(&relativeSrc,&relativeDst,&relativeDefaultDst);
}





static void updateSkeletonAngles(struct skeletonHuman * sk)
{
  unsigned int i=0;
  unsigned int src=0,dst=0;

  double srcD , dstD , srcDefD , dstDefD ;
  for (i=0; i<HUMAN_SKELETON_PARTS; i++)
  {
     src = humanSkeletonJointsRelationMap[i];
     dst = i;

     if ( !skeletonSameJoints(src,dst) )
     {
      srcD = (double) sk->joint[src].x; srcDefD = (double) defaultJoints[src*3+0];
      dstD = (double) sk->joint[dst].x; dstDefD = (double) defaultJoints[dst*3+0];
      sk->relativeJointAngle[i].x=getAngleABCRelative(&srcD,&dstD,&srcDefD,&dstDefD);


      srcD = (double) sk->joint[src].y;
      dstD = (double) sk->joint[dst].y;
      srcDefD = (double) defaultJoints[src*3+1];
      dstDefD = (double) defaultJoints[dst*3+1];
      sk->relativeJointAngle[i].y=getAngleABCRelative(&srcD,&dstD,&srcDefD,&dstDefD);

      srcD = (double) sk->joint[src].z;
      dstD = (double) sk->joint[dst].z;
      srcDefD = (double) defaultJoints[src*3+2];
      dstDefD = (double) defaultJoints[dst*3+2];
      sk->relativeJointAngle[i].z=getAngleABCRelative(&srcD,&dstD,&srcDefD,&dstDefD);
     }

    //sk->relativeJointAngle[i];

  }



}


static double convertSkeletonHumanToSkeletonNAO( struct skeletonNAO * nao , struct skeletonHuman * man)
{
}




static int visualizeSkeletonHuman(const char * filename , struct skeletonHuman * sk)
{
  unsigned int i=0;
  unsigned int src=0,dst=0;
  FILE * fp = fopen(filename,"w");
  if (fp!=0)
  {
    fprintf(fp,"<svg width=\"640\" height=\"480\">\n");
      for (i=0; i<HUMAN_SKELETON_PARTS; i++)
       {
        src = humanSkeletonJointsRelationMap[i];
        dst = i;

        if ( ( !skeletonSameJoints(src,dst) ) && (!skeletonEmptyJoint(sk,src)) && (!skeletonEmptyJoint(sk,dst)) )
        {
         fprintf(fp,"<line x1=\"%0.2f\" y1=\"%0.2f\" x2=\"%0.2f\" y2=\"%0.2f\" style=\"stroke:rgb(255,0,0);stroke-width:2\" />\n",
                 sk->joint2D[src].x, sk->joint2D[src].y , sk->joint2D[dst].x, sk->joint2D[dst].y );
        }
       }

      for (i=0; i<HUMAN_SKELETON_PARTS; i++)
       {
        if (!skeletonEmptyJoint(sk,i))
         {
          fprintf(fp,"<circle cx=\"%0.2f\" cy=\"%0.2f\" r=\"10\" stroke=\"green\" stroke-width=\"4\" fill=\"yellow\" />\n", sk->joint2D[i].x, sk->joint2D[i].y );
          fprintf(fp,"<text x=\"%0.2f\" y=\"%0.2f\">\n",sk->joint2D[i].x-20, sk->joint2D[i].y-20   );
          fprintf(fp,"  <tspan fill=\"red\">%0.2f</tspan>/",sk->relativeJointAngle[i].x);
          fprintf(fp,"  <tspan fill=\"green\">%0.2f</tspan>/",sk->relativeJointAngle[i].y);
          fprintf(fp,"  <tspan fill=\"blue\">%0.2f</tspan>\n",sk->relativeJointAngle[i].z);
          fprintf(fp,"</text>\n"   );
         }

       }


    fprintf(fp,"</svg>\n");

    fclose(fp);
    return 1;
  }
 return 0;
}



#ifdef __cplusplus
}
#endif

#endif // SKELETON_H_INCLUDED

