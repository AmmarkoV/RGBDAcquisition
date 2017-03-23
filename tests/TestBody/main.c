#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../acquisition/Acquisition.h"
#include "../../tools/AmMatrix/matrixCalculations.h"
#include "../../tools/Primitives/jsonCocoSkeleton.h"
#include "../../tools/Primitives/skeleton.h"
#include "../../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/ModelLoader/model_loader_tri.h"
#include "../../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/ModelLoader/model_processor.h"

unsigned int devID=0;
ModuleIdentifier moduleID = TEMPLATE_ACQUISITION_MODULE;//OPENNI1_ACQUISITION_MODULE;//

char inputname[512]= {0};
char modelname[512]= {0};
char outFilename[512]= {0};

void closeEverything()
{
    fprintf(stderr,"Gracefully closing everything .. ");
//Stop our target ( can be network or files or nothing )
    acquisitionStopTargetForFrames(moduleID,devID);
    /*The first argument (Dev ID) could also be ANY_OPENNI2_DEVICE for a single camera setup */
    acquisitionCloseDevice(moduleID,devID);
    acquisitionStopModule(moduleID);

    fprintf(stderr,"Done\n");
    exit(0);
}


int main(int argc, char *argv[])
{
    fprintf(stderr,"TestBody started \n");
    acquisitionRegisterTerminationSignal(&closeEverything);

    snprintf(outFilename,512,"coco.scene");

    unsigned int width=640,height=480,framerate=30;
    unsigned int frameNum=0;
    unsigned int maxFramesToGrab=0;

    unsigned int i=0;
    for (i=0; i<argc; i++)
        {
           if (strcmp(argv[i],"-o")==0)
                {
                   snprintf(outFilename,512,"%s",argv[i+1]);
                }
            if (strcmp(argv[i],"--dev")==0)
                {
                    devID = atoi(argv[i+1]);
                    fprintf(stderr,"Overriding device Used , set to %s ( %u ) \n",argv[i+1],devID);
                }
            else if (
                (strcmp(argv[i],"--from")==0) ||
                (strcmp(argv[i],"-i")==0)
            )
                {
                    strcat(inputname,argv[i+1]);
                    fprintf(stderr,"Input , set to %s  \n",inputname);
                }
            else if (
                (strcmp(argv[i],"--model")==0) ||
                (strcmp(argv[i],"-m")==0)
            )
                {
                    strcat(modelname,argv[i+1]);
                    fprintf(stderr,"Model , set to %s  \n",modelname);
                }
            else if (strcmp(argv[i],"--fps")==0)
                {
                    framerate=atoi(argv[i+1]);
                    fprintf(stderr,"Framerate , set to %u  \n",framerate);
                }
            else if (strcmp(argv[i],"--maxFrames")==0)
                {
                    maxFramesToGrab=atoi(argv[i+1]);
                    fprintf(stderr,"Frame limit , set to %u  \n",maxFramesToGrab);
                }

        }

    if (!acquisitionIsModuleAvailiable(moduleID))
        {
            fprintf(stderr,"The module (%s) you are trying to use is not linked in this build of the Acquisition library..\n",getModuleNameFromModuleID(moduleID));
            return 1;
        }



    struct TRI_Model triModel= {0};
    struct TRI_Model triModelJoints= {0};

    if (strlen(modelname)>0)
        {
            loadModelTri(modelname,&triModel);
            compressTRIModelToJointOnly(&triModelJoints,&triModel);
        }

    FILE *fp;

    fprintf(stderr,"Writing output to %s \n",outFilename);
    fp = fopen(outFilename,"w");

    if (fp!=0)
        {
            fprintf(fp,"%s",
                    "#This is the way to render like the mbv renderer :)\n\
AUTOREFRESH(1500)\n\
BACKGROUND(255,255,255)\n\
\n\
#Bring our world to the MBV coordinate system\n\
SCALE_WORLD(-0.01,-0.01,0.01)\n\
MAP_ROTATIONS(-1,-1,1,zxy)\n\
OFFSET_ROTATIONS(0,0,0)\n\
EMULATE_PROJECTION_MATRIX(535.423889 , 0.0 , 320.000000 , 0.0 , 533.484680 , 240.000000 , 0 , 1) # <-\n\
#MODELVIEW_MATRIX(1,0,0,0, 0,1,0,0 , 0,0,1,0 ,0,0,0,1)\n\
\n\
#MOVE_VIEW(1)\n\
INTERPOLATE_TIME(0)\n\
RATE(1)\n"
                   );


            if (strlen(modelname)>0)
                {
                    fprintf(fp,"OBJECT_TYPE(smartBody,%s)  \n",modelname);
                    fprintf(fp,"RIGID_OBJECT(body,smartBody, 0,0,0,0,0 ,10.0,10.0,10.0 )\n");

                }

            fprintf(fp,"OBJECT_TYPE(objSphere,sphere)  \n");
            for (i=0; i<COCO_PARTS; i++)
                {
                    fprintf(fp,"RIGID_OBJECT(%s,objSphere, 189,3,243,0,0 ,0.59,0.59,0.59 )\n",COCOBodyNames[i]);
                    fprintf(fp,"CONNECTOR(%s,%s, 123,3,0,0,10)\n",COCOBodyNames[i],COCOBodyNames[COCOSkeletonJointsParentRelationMap[i]] );
                }
            fprintf(fp,"\n");

            //We need to initialize our module before calling any related calls to the specific module..
            if (!acquisitionStartModule(moduleID,16 /*maxDevices*/, 0 ))
                {
                    fprintf(stderr,"Could not start module %s ..\n",getModuleNameFromModuleID(moduleID));
                    return 1;
                }

            char * devName = inputname;
            if (strlen(inputname)<1)
                {
                    devName=0;
                }
            if (!acquisitionOpenDevice(moduleID,devID,devName,width,height,framerate))
                {
                    fprintf(stderr,"Could not open device %u ( %s ) of module %s  ..\n",devID,devName,getModuleNameFromModuleID(moduleID));
                    return 1;
                }

            if (maxFramesToGrab==0)
              { maxFramesToGrab=acquisitionGetTotalFrameNumber(moduleID,devID); }
            struct skeletonCOCO skel= {0};



            while  ( (maxFramesToGrab==0)||(frameNum<maxFramesToGrab) )
                {
                    acquisitionStartTimer(0);

                    acquisitionSnapFrames(moduleID,devID);

                    // acquisitionPassFramesToTarget(moduleID,devID,frameNum,0);
                    fprintf(fp,"FRAME(%u)\n",frameNum);

                    char outfilename[1024]= {0};
                    sprintf(outfilename,"frames/%s/dnnOut/colorFrame_%u_%05u.json",inputname,devID,frameNum);
                    fprintf(stderr," will read %s \n",outfilename);
                    //acquisitionSaveColorFrame(moduleID,devID,outfilename);
                    parseJsonCOCOSkeleton(outfilename,&skel);


                    unsigned int i=0;
                    for(i=0; i<COCO_PARTS; i++)
                        {
                            acquisitionGetDepth3DPointAtXYCameraSpace( moduleID,devID,
                                    (unsigned int) skel.joint2D[i].x,
                                    (unsigned int) skel.joint2D[i].y,
                                    &skel.joint[i].x,
                                    &skel.joint[i].y,
                                    &skel.joint[i].z
                                                                     );

                            fprintf(fp,"MOVE(%s,%u,%0.2f,%0.2f,%0.2f,0.0,0.0,0.0,1.0)\n",COCOBodyNames[i],frameNum,skel.joint[i].x,skel.joint[i].y,skel.joint[i].z);
                        }


                    if (strlen(modelname)>0)
                        {
                            float x,y,z,qX,qY,qZ,qW;
                            //convertCOCO_To_Smartbody_TRI(&skel,&triModel,&x,&y,&z,&qX,&qY,&qZ,&qW); //triModelJoints
                            //fprintf(fp,"MOVE(body,%u,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f)\n",frameNum,x,y,z,qX,qY,qZ,qW);
                        }

                    fprintf(fp,"\n");

                    acquisitionStopTimer(0);
                    if (frameNum%25==0) fprintf(stderr,"%0.2f fps\n",acquisitionGetTimerFPS(0));
                    ++frameNum;


                }

            fprintf(stderr,"Done grabbing %u frames! \n",maxFramesToGrab);
            fclose(fp);


            if (strlen(modelname)>0)
                {
                    deallocInternalsOfModelTri(&triModel);
                }

        }



    closeEverything();

    return 0;
}
