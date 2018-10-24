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



int main(int argc, char **argv)
{
    struct BVH_MotionCapture bvhMotion={0};

    bvh_loadBVH("Motions/example.bvh", &bvhMotion);

    bvh_printBVH(&bvhMotion);

    //Test getting rotations for a joint..
    BVHFrameID frameID = 0;
    BVHJointID jID=0;
    if ( bvh_getJointIDFromJointName(&bvhMotion ,"RightFoot",&jID) )
    {
      for (frameID=0; frameID<bvhMotion.numberOfFrames; frameID++)
       {
         fprintf(stderr,"Joint %s \n",bvhMotion.jointHierarchy[jID].jointName);
         fprintf(stderr,"XRotation:%0.2f ",bvh_getJointRotationXAtFrame(&bvhMotion , jID ,  frameID));
         fprintf(stderr,"YRotation:%0.2f ",bvh_getJointRotationYAtFrame(&bvhMotion , jID ,  frameID));
         fprintf(stderr,"ZRotation:%0.2f\n",bvh_getJointRotationZAtFrame(&bvhMotion , jID ,  frameID));
       }
    }

    dumpBVHToTrajectoryParser("Scenes/bvh.conf",&bvhMotion);

    return 0;
}
