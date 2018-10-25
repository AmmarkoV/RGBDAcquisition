/** @file main.c
 *  @brief  A minimal binary that parses BVH scenes using OGLRendererSandbox s
 *          X86 compilation: gcc -o -L/usr/X11/lib   main main.c -lGL -lX11 -lpng -ljpeg
 *          X64 compilation: gcc -o -L/usr/X11/lib64 main main.c -lGL -lX11 -lpng -ljpeg
 *  @author Ammar Qammaz (AmmarkoV)
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "../../Library/TrajectoryParser/TrajectoryParserDataStructures.h"
#include "../../Library/MotionCaptureLoader/bvh_loader.h"
#include "../../Library/MotionCaptureLoader/bvh_to_trajectoryParser.h"

void testPrintout(struct BVH_MotionCapture * bvhMotion,const char * jointName)
{
    //Test getting rotations for a joint..
    BVHFrameID frameID = 0;
    BVHJointID jID=0;
   fprintf(stderr,"\nJoint %s \n",bvhMotion->jointHierarchy[jID].jointName);
    if ( bvh_getJointIDFromJointName(bvhMotion ,jointName,&jID) )
    {
      for (frameID=0; frameID<bvhMotion->numberOfFrames; frameID++)
       {
         fprintf(stderr,"Frame %u \n",frameID);
         fprintf(stderr,"XRotation:%0.2f ",bvh_getJointRotationXAtFrame(bvhMotion , jID ,  frameID));
         fprintf(stderr,"YRotation:%0.2f ",bvh_getJointRotationYAtFrame(bvhMotion , jID ,  frameID));
         fprintf(stderr,"ZRotation:%0.2f\n",bvh_getJointRotationZAtFrame(bvhMotion , jID ,  frameID));
       }
    }
}



int main(int argc, char **argv)
{
    const char * fromBVHFile="Motions/example.bvh";
    const char * toSceneFile="Scenes/bvh.conf";


    unsigned int i=0;
    for (i=0; i<argc; i++)
    {
        if (strcmp(argv[i],"--from")==0)
        {
          fromBVHFile=argv[i+1];
        } else
        if (strcmp(argv[i],"--to")==0)
        {
          toSceneFile=argv[i+1];
        }
    }

    struct BVH_MotionCapture bvhMotion={0};

    bvh_loadBVH(fromBVHFile, &bvhMotion);

    //Test printout of all rotations of a specific joint..
    //testPrintout(&bvhMotion,"RightFoot");

    bvh_printBVH(&bvhMotion);


    dumpBVHToTrajectoryParser(toSceneFile,&bvhMotion);

    bvh_free(&bvhMotion);

    return 0;
}
