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
#include "../../Library/MotionCaptureLoader/bvh_to_tri_pose.h"
#include "../../Library/MotionCaptureLoader/bvh_to_svg.h"

void testPrintout(struct BVH_MotionCapture * bvhMotion,const char * jointName)
{
    //Test getting rotations for a joint..
    BVHFrameID frameID = 0;
    BVHJointID jID=0;
    if ( bvh_getJointIDFromJointName(bvhMotion ,jointName,&jID) )
    {
      fprintf(stderr,"\nJoint %s (#%u) \n",bvhMotion->jointHierarchy[jID].jointName,jID);

      fprintf(
              stderr,"Channels ( %u,%u,%u )\n",
              bvhMotion->jointToMotionLookup[jID].channelIDMotionOffset[BVH_ROTATION_X],
              bvhMotion->jointToMotionLookup[jID].channelIDMotionOffset[BVH_ROTATION_Y],
              bvhMotion->jointToMotionLookup[jID].channelIDMotionOffset[BVH_ROTATION_Z]
             );

       for (frameID=0; frameID<bvhMotion->numberOfFrames; frameID++)
       {

         fprintf(stderr,"Frame %u \n",frameID);
         fprintf(stderr,"XRotation:%0.2f " ,bvh_getJointRotationXAtFrame(bvhMotion , jID ,  frameID));
         fprintf(stderr,"YRotation:%0.2f " ,bvh_getJointRotationYAtFrame(bvhMotion , jID ,  frameID));
         fprintf(stderr,"ZRotation:%0.2f\n",bvh_getJointRotationZAtFrame(bvhMotion , jID ,  frameID));
       }
    }
}

void incorrectArguments()
{
  fprintf(stderr,"Incorrect number of arguments.. \n");
  exit(1);
}


int main(int argc, char **argv)
{
    const char * fromBVHFile="Motions/example.bvh";
    const char * toSceneFile="Scenes/bvh.conf";
    const char * toSceneFileTRI="Scenes/bvhTRI.conf";
    const char * toSVGDirectory="tmp/";
    unsigned int convertToSVG=0;
    unsigned int convertToCSV=0;
    unsigned int maxFrames = 0;
    unsigned int onlyFirstFrame=0;
    unsigned int usePosition=0;

    float scaleWorld=1.0;
    float cameraPositionOffset[3]={0};
    float cameraRotationOffset[3]={0};
    float objectRotationOffset[3]={0};

    unsigned int i=0;
    for (i=0; i<argc; i++)
    {
        if (strcmp(argv[i],"--maxFrames")==0)
        {
          if (i+1>=argc)  { incorrectArguments();}
          maxFrames=atoi(argv[i+1]);
        } else
        if (strcmp(argv[i],"--from")==0)
        {
          if (i+1>=argc)  { incorrectArguments();}
          fromBVHFile=argv[i+1];
        } else
        if (strcmp(argv[i],"--to")==0)
        {
          if (i+1>=argc)  { incorrectArguments(); }
          toSceneFile=argv[i+1];
        } else
        if (strcmp(argv[i],"--csv")==0)
        {
          if (i+1>=argc)  { incorrectArguments(); }
          toSVGDirectory=argv[i+1];
          convertToCSV=1;
        } else
        if (strcmp(argv[i],"--svg")==0)
        {
          if (i+1>=argc)  { incorrectArguments(); }
          toSVGDirectory=argv[i+1];

          char removeOldSVGFilesCommand[512];
          snprintf(removeOldSVGFilesCommand,512,"rm %s/*.svg",toSVGDirectory);
          int res = system(removeOldSVGFilesCommand);
          if (res!=0) { fprintf(stderr,"Could not clean svg files in %s",toSVGDirectory); }
          convertToSVG=1;
        } else
        if (strcmp(argv[i],"--onlyFirstFrame")==0)
        {
          onlyFirstFrame=1;
        } else
        if (strcmp(argv[i],"--usePosition")==0)
        {
          usePosition=1;
        } else
        if (strcmp(argv[i],"--scale")==0)
        {
          if (i+1>=argc)  { incorrectArguments(); }
          scaleWorld=atof(argv[i+1]);
          //TODO: Use this..
        } else
        if (strcmp(argv[i],"--cameraPosition")==0)
        {
          if (i+3>=argc)  { incorrectArguments(); }
          cameraPositionOffset[0]=-1*atof(argv[i+1])/10;
          cameraPositionOffset[1]=atof(argv[i+2])/10;
          cameraPositionOffset[2]=atof(argv[i+3])/10;
        } else
        if (strcmp(argv[i],"--cameraRotation")==0)
        {
          if (i+3>=argc)  { incorrectArguments(); }
          cameraRotationOffset[0]=atof(argv[i+1]);
          cameraRotationOffset[1]=atof(argv[i+2]);
          cameraRotationOffset[2]=atof(argv[i+3]);
        } else
        if (strcmp(argv[i],"--objectRotation")==0)
        {
          if (i+3>=argc)  { incorrectArguments(); }
          objectRotationOffset[0]=atof(argv[i+1]);
          objectRotationOffset[1]=atof(argv[i+2]);
          objectRotationOffset[2]=atof(argv[i+3]);
        }
    }

    struct BVH_MotionCapture bvhMotion={0};

    //First of all we need to load the BVH file
    bvh_loadBVH(fromBVHFile, &bvhMotion, scaleWorld);

    //Change joint names..
    bvh_renameJoints(&bvhMotion);

    //We can limit the number of frames
    if (maxFrames!=0)
      {
        //Only reducing number of frames
        if (bvhMotion.numberOfFrames>maxFrames)
        {
          bvhMotion.numberOfFrames = maxFrames;
        }
      }

    //SVG or CSV output ..
    if ( (convertToSVG) || (convertToCSV) )
    {
     dumpBVHToSVG(
                  toSVGDirectory,
                  convertToSVG,
                  convertToCSV,
                  &bvhMotion,
                  640,
                  480,
                  cameraPositionOffset,
                  cameraRotationOffset,
                  objectRotationOffset,
                  //Don't care about randomization
                  0,0,0,0,0
                 );
      return 0;
    }




    bvh_printBVH(&bvhMotion);
    //bvh_printBVHJointToMotionLookupTable(&bvhMotion);

    struct bvhToTRI bvhtri={0};
    bvh_loadBVHToTRI("Motions/cmu.profile",&bvhtri,&bvhMotion);

    if (onlyFirstFrame)
    {
     bvh_copyMotionFrame(&bvhMotion, 0, 1 );
     bvhMotion.numberOfFrames=2; //Just Render one frame..
    }



    dumpBVHToTrajectoryParserTRI(toSceneFileTRI,&bvhMotion,&bvhtri,usePosition,0);
    dumpBVHToTrajectoryParser(toSceneFile,&bvhMotion);

    //Test printout of all rotations of a specific joint..
    testPrintout(&bvhMotion,"rknee");


    bvh_free(&bvhMotion);

    return 0;
}
