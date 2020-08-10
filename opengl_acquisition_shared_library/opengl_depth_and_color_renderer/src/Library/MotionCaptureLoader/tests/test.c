#include "test.h"

#include <stdio.h>
#include "../edit/bvh_remapangles.h"

void testPrintout(struct BVH_MotionCapture * bvhMotion,const char * jointName)
{
    //Test getting rotations for a joint..
    BVHJointID jID=0;
    //if ( bvh_getJointIDFromJointName(bvhMotion ,jointName,&jID) )
    if ( bvh_getJointIDFromJointNameNocase(bvhMotion ,jointName,&jID) )
    {
      fprintf(stderr,"\nJoint %s (#%u) \n",bvhMotion->jointHierarchy[jID].jointName,jID);

      fprintf(
              stderr,"Channels ( %u,%u,%u )\n",
              bvhMotion->jointToMotionLookup[jID].channelIDMotionOffset[BVH_ROTATION_X],
              bvhMotion->jointToMotionLookup[jID].channelIDMotionOffset[BVH_ROTATION_Y],
              bvhMotion->jointToMotionLookup[jID].channelIDMotionOffset[BVH_ROTATION_Z]
             );

       BVHFrameID frameID = 0;
       for (frameID=0; frameID<bvhMotion->numberOfFrames; frameID++)
       {

         fprintf(stderr,"Frame %u \n",frameID);
         fprintf(stderr,"XRotation:%0.2f " ,bvh_getJointRotationXAtFrame(bvhMotion , jID ,  frameID));
         fprintf(stderr,"YRotation:%0.2f " ,bvh_getJointRotationYAtFrame(bvhMotion , jID ,  frameID));
         fprintf(stderr,"ZRotation:%0.2f\n",bvh_getJointRotationZAtFrame(bvhMotion , jID ,  frameID));
       }
    }
}


int bvh_testConstrainRotations()
{
  /*
  fprintf(stderr,"Testing bvh_rotation constraint\n");
  unsigned int i=0;
  float angle = -720;
  for (i=0; i<1440; i++)
  {
    fprintf(stderr,"| Angle:%0.2f | Centered at 0:%0.2f | Flipped at 0:%0.2f\n", //| Centered at 180:%0.2f
    angle,
    bvh_constrainAngleCentered0(angle,0),
    bvh_constrainAngleCentered0(angle,1)
    //bvh_constrainAngleCentered180(angle)
    );
    angle=angle+1.0;
  }

  */


  fprintf(stderr,"Testing bvh_rotation front constraint\n");
  unsigned int i=0;
  float angle = -45;
  for (i=0; i<90; i++)
  {
    fprintf(stderr,"| Angle:%0.2f | Front Centered at 0 :%0.2f\n", //| Centered at 180:%0.2f
    angle,
    bvh_RemapAngleCentered0(angle,BVH_ENFORCE_FRONT_ORIENTATION)
    );
    angle=angle+1.0;
  }


  fprintf(stderr,"Testing bvh_rotation back constraint\n");
  i=0;
  angle = 135;
  for (i=0; i<90; i++)
  {
    fprintf(stderr,"| Angle:%0.2f | Back Centered at 0 :%0.2f\n", //| Centered at 180:%0.2f
    angle,
    bvh_RemapAngleCentered0(angle,BVH_ENFORCE_BACK_ORIENTATION)
    );
    angle=angle+1.0;
  }


  fprintf(stderr,"Testing bvh_rotation right constraint\n");
  i=0;
  angle = 45;
  for (i=0; i<90; i++)
  {
    fprintf(stderr,"| Angle:%0.2f | Right Centered at 0 :%0.2f\n", //| Centered at 180:%0.2f
    angle,
    bvh_RemapAngleCentered0(angle,BVH_ENFORCE_RIGHT_ORIENTATION)
    );
    angle=angle+1.0;
  }


  fprintf(stderr,"Testing bvh_rotation left constraint\n");
  i=0;
  angle = -135;
  for (i=0; i<90; i++)
  {
    fprintf(stderr,"| Angle:%0.2f | Left Centered at 0 :%0.2f\n", //| Centered at 180:%0.2f
    angle,
    bvh_RemapAngleCentered0(angle,BVH_ENFORCE_LEFT_ORIENTATION)
    );
    angle=angle+1.0;
  }


 return 0;
}
