/** @file main.c
 *  @brief  A minimal binary that parses BVH scenes using OGLRendererSandbox s
 *          X86 compilation: gcc -o -L/usr/X11/lib   main main.c -lGL -lX11 -lpng -ljpeg
 *          X64 compilation: gcc -o -L/usr/X11/lib64 main main.c -lGL -lX11 -lpng -ljpeg
 *  @author Ammar Qammaz (AmmarkoV)
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "../../Library/TrajectoryParser/TrajectoryParserDataStructures.h"
#include "../../Library/MotionCaptureLoader/bvh_loader.h"
#include "../../Library/MotionCaptureLoader/bvh_to_tri_pose.h"

#include "../../Library/MotionCaptureLoader/export/bvh_to_trajectoryParserTRI.h"
#include "../../Library/MotionCaptureLoader/export/bvh_to_trajectoryParserPrimitives.h"
#include "../../Library/MotionCaptureLoader/export/bvh_export.h"
#include "../../Library/MotionCaptureLoader/export/bvh_to_bvh.h"

#include "../../Library/MotionCaptureLoader/edit/bvh_inverseKinematics.h"

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
    const char * toCSVFilename="data.csv";
    unsigned int convertToSVG=0;
    unsigned int convertToCSV=0;
    unsigned int maxFrames = 0;
    unsigned int occlusions = 0;
    float scaleWorld=1.0;
    unsigned int flipOrientation = 0;
    unsigned int flipRandomizationOrientation = 0;

    struct BVH_MotionCapture bvhMotion={0};

    unsigned int i=0;
    for (i=0; i<argc; i++)
    {
        //-----------------------------------------------------
        if (strcmp(argv[i],"--test")==0)
        {
          bvh_testConstrainRotations();
          exit(0);
        } else
        //-----------------------------------------------------
        if (strcmp(argv[i],"--occlusions")==0)
        {
          occlusions=1;
        } else
        //-----------------------------------------------------
        if (strcmp(argv[i],"--print")==0)
        {
          bvh_printBVH(&bvhMotion);
        } else
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
          bvh_ConstrainRotations(&bvhMotion,flipOrientation);
        } else
        //-----------------------------------------------------
        if (strcmp(argv[i],"--to")==0)
        {
          if (i+1>=argc)  { incorrectArguments(); }
          toSceneFile=argv[i+1];
           struct bvhToTRI bvhtri={0};
           bvh_loadBVHToTRI("Motions/cmu.profile",&bvhtri,&bvhMotion);
           dumpBVHToTrajectoryParserTRI(toSceneFileTRI,&bvhMotion,&bvhtri,1/*USE Irugubak oisutuib*/,0);
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
          toCSVFilename=argv[i+2];
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
        if (strcmp(argv[i],"--swap")==0)
        {
          if (i+2>=argc)  { incorrectArguments(); }
          bvh_GrowMocapFileBySwappingJointAndItsChildren(
                                                          &bvhMotion,
                                                          argv[i+1],
                                                          argv[i+2],
                                                          0
                                                         );
        } else
        //-----------------------------------------------------


        //-----------------------------------------------------
        if (strcmp(argv[i],"--mirror")==0)
        {
          if (i+2>=argc)  { incorrectArguments(); }
          bvh_MirrorJointsThroughIK(
                                    &bvhMotion,
                                    argv[i+1],
                                    argv[i+2]
                                  );
        } else
        //-----------------------------------------------------

        //-----------------------------------------------------
        if (strcmp(argv[i],"--interpolate")==0)
        {
          bvh_InterpolateMotion(
                                &bvhMotion
                               );
        } else
        //-----------------------------------------------------

        //-----------------------------------------------------
        if (strcmp(argv[i],"--repeat")==0)
        {
          if (i+1>=argc)  { incorrectArguments(); }
          bvh_GrowMocapFileByCopyingExistingMotions(
                                                     &bvhMotion,
                                                     atoi(argv[i+1])
                                                   );
        } else
        //-----------------------------------------------------
        if (strcmp(argv[i],"--setPositionRotation")==0)
        {
          if (i+6>=argc)  { incorrectArguments(); }
          float cameraPositionOffset[3];
          float cameraRotationOffset[3];

          cameraPositionOffset[0]=-1*atof(argv[i+1])/10;
          cameraPositionOffset[1]=-1*atof(argv[i+2])/10;
          cameraPositionOffset[2]=-1*atof(argv[i+3])/10;
          cameraRotationOffset[0]=atof(argv[i+4]);
          cameraRotationOffset[1]=atof(argv[i+5]);
          cameraRotationOffset[2]=atof(argv[i+6]);
          bvh_SetPositionRotation(
                                  &bvhMotion,
                                  cameraPositionOffset,
                                  cameraRotationOffset
                                 );
          bvh_ConstrainRotations(&bvhMotion,flipOrientation);
        } else
        //-----------------------------------------------------
        if (strcmp(argv[i],"--offsetPositionRotation")==0)
        {
          if (i+6>=argc)  { incorrectArguments(); }
          float cameraPositionOffset[3];
          float cameraRotationOffset[3];

          cameraPositionOffset[0]=-1*atof(argv[i+1])/10;
          cameraPositionOffset[1]=-1*atof(argv[i+2])/10;
          cameraPositionOffset[2]=-1*atof(argv[i+3])/10;
          cameraRotationOffset[0]=atof(argv[i+4]);
          cameraRotationOffset[1]=atof(argv[i+5]);
          cameraRotationOffset[2]=atof(argv[i+6]);
          bvh_OffsetPositionRotation(
                                     &bvhMotion,
                                     cameraPositionOffset,
                                     cameraRotationOffset
                                    );
          bvh_ConstrainRotations(&bvhMotion,flipOrientation);
        } else
        //-----------------------------------------------------
        if (strcmp(argv[i],"--flipRandomizationOrientation")==0)
        {
             flipRandomizationOrientation=1;
        } else
        //-----------------------------------------------------
        if (strcmp(argv[i],"--randomize")==0)
        {
          if (i+12>=argc)  { incorrectArguments(); }
          srand(time(NULL));

          float minimumPosition[3];
          float minimumRotation[3];
          float maximumPosition[3];
          float maximumRotation[3];

          //----
          minimumPosition[0]=-1*atof(argv[i+1])/10;
          minimumPosition[1]=-1*atof(argv[i+2])/10;
          minimumPosition[2]=-1*atof(argv[i+3])/10;
          //----
          minimumRotation[0]=atof(argv[i+4]);
          minimumRotation[1]=atof(argv[i+5]);
          minimumRotation[2]=atof(argv[i+6]);
          //----
          maximumPosition[0]=-1*atof(argv[i+7])/10;
          maximumPosition[1]=-1*atof(argv[i+8])/10;
          maximumPosition[2]=-1*atof(argv[i+9])/10;
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

          bvh_ConstrainRotations(&bvhMotion,flipRandomizationOrientation);
        } else
        //-----------------------------------------------------
        //-----------------------------------------------------
        if (strcmp(argv[i],"--randomizeranges")==0)
        {
          if (i+24>=argc)  { incorrectArguments(); }
          srand(time(NULL));

          float minimumPositionRangeA[3];
          float minimumRotationRangeA[3];
          float maximumPositionRangeA[3];
          float maximumRotationRangeA[3];

          float minimumPositionRangeB[3];
          float minimumRotationRangeB[3];
          float maximumPositionRangeB[3];
          float maximumRotationRangeB[3];

          //----
          minimumPositionRangeA[0]=-1*atof(argv[i+1])/10;
          minimumPositionRangeA[1]=-1*atof(argv[i+2])/10;
          minimumPositionRangeA[2]=-1*atof(argv[i+3])/10;
          //----
          minimumRotationRangeA[0]=atof(argv[i+4]);
          minimumRotationRangeA[1]=atof(argv[i+5]);
          minimumRotationRangeA[2]=atof(argv[i+6]);
          //----
          maximumPositionRangeA[0]=-1*atof(argv[i+7])/10;
          maximumPositionRangeA[1]=-1*atof(argv[i+8])/10;
          maximumPositionRangeA[2]=-1*atof(argv[i+9])/10;
          //----
          maximumRotationRangeA[0]=atof(argv[i+10]);
          maximumRotationRangeA[1]=atof(argv[i+11]);
          maximumRotationRangeA[2]=atof(argv[i+12]);

          //----
          minimumPositionRangeB[0]=-1*atof(argv[i+13])/10;
          minimumPositionRangeB[1]=-1*atof(argv[i+14])/10;
          minimumPositionRangeB[2]=-1*atof(argv[i+15])/10;
          //----
          minimumRotationRangeB[0]=atof(argv[i+16]);
          minimumRotationRangeB[1]=atof(argv[i+17]);
          minimumRotationRangeB[2]=atof(argv[i+18]);
          //----
          maximumPositionRangeB[0]=-1*atof(argv[i+19])/10;
          maximumPositionRangeB[1]=-1*atof(argv[i+20])/10;
          maximumPositionRangeB[2]=-1*atof(argv[i+21])/10;
          //----
          maximumRotationRangeB[0]=atof(argv[i+22]);
          maximumRotationRangeB[1]=atof(argv[i+23]);
          maximumRotationRangeB[2]=atof(argv[i+24]);

          bvh_RandomizePositionRotation2Ranges(
                                                &bvhMotion,
                                                minimumPositionRangeA,
                                                minimumRotationRangeA,
                                                maximumPositionRangeA,
                                                maximumRotationRangeA,
                                                minimumPositionRangeB,
                                                minimumRotationRangeB,
                                                maximumPositionRangeB,
                                                maximumRotationRangeB
                                              );

          bvh_ConstrainRotations(&bvhMotion,flipRandomizationOrientation);
        }
    }






    //SVG or CSV output ..
    if ( (convertToSVG) || (convertToCSV) )
    {
     dumpBVHToSVGCSV(
                     toSVGDirectory,
                     toCSVFilename,
                     convertToSVG,
                     convertToCSV,
                     &bvhMotion,
                     //640,480 , 575.57 , 575.57, //Kinect
                     1920, 1080, 582.18394,   582.52915, // https://gopro.com/help/articles/Question_Answer/HERO4-Field-of-View-FOV-Information
                     occlusions,
                     1,//Filter out all poses where even one joint is behind camera
                     1,//Filter out all poses where even one joint is outside of 2D frame
                     1,//Filter top left weird random skelingtons ( skeletons )
                     0//We don't want to convert to radians
                 );
      return 0;
    }


    bvh_free(&bvhMotion);

    return 0;
}
