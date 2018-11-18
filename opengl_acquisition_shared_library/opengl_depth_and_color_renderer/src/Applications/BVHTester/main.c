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
#include "../../Library/MotionCaptureLoader/bvh_to_tri_pose.h"
#include "../../Library/MotionCaptureLoader/export/bvh_to_trajectoryParserTRI.h"
#include "../../Library/MotionCaptureLoader/export/bvh_to_trajectoryParserPrimitives.h"
#include "../../Library/MotionCaptureLoader/export/bvh_export.h"
#include "../../Library/MotionCaptureLoader/export/bvh_to_bvh.h"

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
    const char * toBVHFile="Motions/bvh.bvh";
    const char * toSceneFile="Scenes/bvh.conf";
    const char * toSceneFileTRI="Scenes/bvhTRI.conf";
    const char * toSVGDirectory="tmp/";
    unsigned int convertToSVG=0;
    unsigned int convertToCSV=0;
    unsigned int maxFrames = 0;
    unsigned int useOriginalPosition=0;

    float scaleWorld=1.0;
    float cameraPositionOffset[3]={0};
    float cameraRotationOffset[3]={0};
    float objectRotationOffset[3]={0};


    struct BVH_MotionCapture bvhMotion={0};

    unsigned int i=0;
    for (i=0; i<argc; i++)
    {
        //-----------------------------------------------------
        if (strcmp(argv[i],"--scale")==0)
        {
          if (i+1>=argc)  { incorrectArguments(); }
          scaleWorld=atof(argv[i+1]);
        } else
        //-----------------------------------------------------
        if (strcmp(argv[i],"--onlyFirstFrame")==0)
        {
          bvh_copyMotionFrame(&bvhMotion, 0, 1 );
          bvhMotion.numberOfFrames=2; //Just Render one frame..
        } else
        //-----------------------------------------------------
        if (strcmp(argv[i],"--maxFrames")==0)
        {
          if (i+1>=argc)  { incorrectArguments();}
          maxFrames=atoi(argv[i+1]);
          //We can limit the number of frames
          if (maxFrames!=0)
            {
             //Only reducing number of frames
             if (bvhMotion.numberOfFrames>maxFrames)
               {
                 bvhMotion.numberOfFrames = maxFrames;
               }
            }
        } else
        //-----------------------------------------------------
        if (strcmp(argv[i],"--from")==0)
        {
          if (i+1>=argc)  { incorrectArguments();}
          fromBVHFile=argv[i+1];
          //First of all we need to load the BVH file
          bvh_loadBVH(fromBVHFile, &bvhMotion, scaleWorld);
          //Change joint names..
          bvh_renameJointsForCompatibility(&bvhMotion);
        } else
        //-----------------------------------------------------
        if (strcmp(argv[i],"--print")==0)
        {
          bvh_printBVH(&bvhMotion);
        } else
        if (strcmp(argv[i],"--to")==0)
        {
          if (i+1>=argc)  { incorrectArguments(); }
          toSceneFile=argv[i+1];
           struct bvhToTRI bvhtri={0};
           bvh_loadBVHToTRI("Motions/cmu.profile",&bvhtri,&bvhMotion);
           dumpBVHToTrajectoryParserTRI(toSceneFileTRI,&bvhMotion,&bvhtri,useOriginalPosition,0);
           dumpBVHToTrajectoryParserPrimitives(toSceneFile,&bvhMotion);
        } else
        //-----------------------------------------------------
        if (strcmp(argv[i],"--bvh")==0)
        {
          if (i+1>=argc)  { incorrectArguments(); }
          toBVHFile=argv[i+1];
          dumpBVHToBVH(
                        toBVHFile,
                        &bvhMotion
                      );
        } else
        //-----------------------------------------------------
        if (strcmp(argv[i],"--csv")==0)
        {
          if (i+1>=argc)  { incorrectArguments(); }
          toSVGDirectory=argv[i+1];
          convertToCSV=1;
        } else
        //-----------------------------------------------------
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
        //-----------------------------------------------------
        if (strcmp(argv[i],"--repeat")==0)
        {
          bvh_GrowMocapFileByCopyingExistingMotions(
                                                     &bvhMotion,
                                                     atoi(argv[i+1])
                                                   );
        } else
        //-----------------------------------------------------
        if (strcmp(argv[i],"--setPositionRotation")==0)
        {
          cameraPositionOffset[0]=-1*atof(argv[i+1])/10;
          cameraPositionOffset[1]=atof(argv[i+2])/10;
          cameraPositionOffset[2]=atof(argv[i+3])/10;
          cameraRotationOffset[0]=atof(argv[i+1]);
          cameraRotationOffset[1]=atof(argv[i+2]);
          cameraRotationOffset[2]=atof(argv[i+3]);
          bvh_SetPositionRotation(
                                  &bvhMotion,
                                  cameraPositionOffset,
                                  cameraRotationOffset
                                 );
        } else
        //-----------------------------------------------------
        if (strcmp(argv[i],"--randomize")==0)
        {
          if (i+12>=argc)  { incorrectArguments(); }
          float minimumPosition[3];
          float minimumRotation[3];
          float maximumPosition[3];
          float maximumRotation[3];

          //----
          minimumPosition[0]=-1*atof(argv[i+1])/10;
          minimumPosition[1]=atof(argv[i+2])/10;
          minimumPosition[2]=atof(argv[i+3])/10;
          //----
          minimumRotation[0]=atof(argv[i+4]);
          minimumRotation[1]=atof(argv[i+5]);
          minimumRotation[2]=atof(argv[i+6]);
          //----
          maximumPosition[0]=-1*atof(argv[i+7])/10;
          maximumPosition[1]=atof(argv[i+8])/10;
          maximumPosition[2]=atof(argv[i+9])/10;
          //----
          maximumRotation[0]=atof(argv[i+10]);
          maximumRotation[1]=atof(argv[i+11]);
          maximumRotation[2]=atof(argv[i+12]);


          bvh_RandomizePositionRotation(
                                         &bvhMotion,
                                         minimumPosition,
                                         minimumRotation,
                                         maximumPosition,
                                         maximumRotation
                                       );
        } else
        //-----------------------------------------------------
        if (strcmp(argv[i],"--useOriginalPosition")==0)
        {
          useOriginalPosition=1;
        } else
        //-----------------------------------------------------
        if (strcmp(argv[i],"--cameraPosition")==0)
        {
          if (i+3>=argc)  { incorrectArguments(); }
          cameraPositionOffset[0]=-1*atof(argv[i+1])/10;
          cameraPositionOffset[1]=atof(argv[i+2])/10;
          cameraPositionOffset[2]=atof(argv[i+3])/10;
        } else
        //-----------------------------------------------------
        if (strcmp(argv[i],"--cameraRotation")==0)
        {
          if (i+3>=argc)  { incorrectArguments(); }
          cameraRotationOffset[0]=atof(argv[i+1]);
          cameraRotationOffset[1]=atof(argv[i+2]);
          cameraRotationOffset[2]=atof(argv[i+3]);
        } else
        //-----------------------------------------------------
        if (strcmp(argv[i],"--objectRotation")==0)
        {
          if (i+3>=argc)  { incorrectArguments(); }
          objectRotationOffset[0]=atof(argv[i+1]);
          objectRotationOffset[1]=atof(argv[i+2]);
          objectRotationOffset[2]=atof(argv[i+3]);
        }
        //-----------------------------------------------------
    }






    //SVG or CSV output ..
    if ( (convertToSVG) || (convertToCSV) )
    {
     dumpBVHToSVGCSV(
                     toSVGDirectory,
                     convertToSVG,
                     convertToCSV,
                     &bvhMotion,
                     640,
                     480,
                     useOriginalPosition,
                     cameraPositionOffset,
                     cameraRotationOffset,
                     objectRotationOffset,
                     //Don't care about randomization
                     0,0,0,0,0,
                     //
                     1,//Filter out all poses where even one joint is behind camera
                     1,//Filter out all poses where even one joint is outside of 2D frame
                     0//We don't want to convert to radians
                 );
      return 0;
    }


    bvh_free(&bvhMotion);

    return 0;
}
