#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "../acquisition/Acquisition.h"
#include "../acquisitionSegment/AcquisitionSegment.h"


int calibrationSet = 0;
char calibrationFile[2048]={0};
struct calibration calib;

char outputfoldername[512]={0};
char inputname[512]={0};

int makepath(char * path)
{
    //FILE *fp;
    /* Open the command for reading. */
    char command[1024];
    sprintf(command,"mkdir -p %s",outputfoldername);
    fprintf(stderr,"Executing .. %s \n",command);

    return system(command);
}

unsigned char * copyRGB(unsigned char * source , unsigned int width , unsigned int height)
{
  unsigned char * output = (unsigned char*) malloc(width*height*3*sizeof(unsigned char));
  if (output==0) { return 0; }
  memcpy(output , source , width*height*3*sizeof(unsigned char));
  return output;
}

unsigned short * copyDepth(unsigned short * source , unsigned int width , unsigned int height)
{
  unsigned short * output = (unsigned short*) malloc(width*height*sizeof(unsigned short));
  if (output==0) { return 0; }
  memcpy(output , source , width*height*sizeof(unsigned short));
  return output;
}


int pickCombinationModeFromString(char * str)
{
  if (strcasecmp(str,"and")==0) { return COMBINE_AND; } else
  if (strcasecmp(str,"or")==0)  { return COMBINE_OR; } else
  if (strcasecmp(str,"xor")==0) { return COMBINE_XOR; } else
  if (strcasecmp(str,"rgb")==0) { return COMBINE_KEEP_ONLY_RGB; } else
  if (strcasecmp(str,"depth")==0) { return COMBINE_KEEP_ONLY_DEPTH; }

  fprintf(stderr,"Could not understand combination method %s\n",str);
  return DONT_COMBINE;
}



int main(int argc, char *argv[])
{
 fprintf(stderr,"Generic Multiplexed Grabber Application based on Acquisition lib .. \n");
 unsigned int possibleModules = acquisitionGetModulesCount();
 fprintf(stderr,"Linked to %u modules.. \n",possibleModules);

 if (possibleModules==0)
    {
       fprintf(stderr,"Acquisition Library is linked to zero modules , can't possibly do anything..\n");
       return 1;
    }

 //We want to grab multiple frames in this example if the user doesnt supply a parameter default is 10..
  unsigned int frameNum=0,maxFramesToGrab=10;
  ModuleIdentifier moduleID_1 = TEMPLATE_ACQUISITION_MODULE;//OPENNI1_ACQUISITION_MODULE;//

  /*! --------------------------------- INITIALIZATION FROM COMMAND LINE PARAMETERS --------------------------------- */


   //------------------------------------------------------------------
   //                        CONFIGURATION
   //------------------------------------------------------------------
   int doNotSegmentRGB=1;
   int doNotSegmentDepth=1;


   int combinationMode=DONT_COMBINE;


   struct SegmentationFeaturesRGB segConfRGB={0};

   segConfRGB.floodErase.totalPoints = 0;

   segConfRGB.minX=0;  segConfRGB.maxX=640;
   segConfRGB.minY=0; segConfRGB.maxY=480;

   segConfRGB.minR=0; segConfRGB.minG=0; segConfRGB.minB=0;
   segConfRGB.maxR=256; segConfRGB.maxG=256; segConfRGB.maxB=256;

   segConfRGB.enableReplacingColors=0;
   segConfRGB.replaceR=92; segConfRGB.replaceG=45; segConfRGB.replaceB=36;

   struct SegmentationFeaturesDepth segConfDepth={0};
   segConfDepth.minX=0;  segConfDepth.maxX=640;
   segConfDepth.minY=0; segConfDepth.maxY=480;
   segConfDepth.minDepth=0; segConfDepth.maxDepth=32500;
   //------------------------------------------------------------------
   //------------------------------------------------------------------




  int i=0;
  for (i=0; i<argc; i++)
  {
    if (strcmp(argv[i],"-calibration")==0) {
                                             calibrationSet=1;
                                             strncpy(calibrationFile,argv[i+1],2048);
                                           } else
    if (strcmp(argv[i],"-maxFrames")==0) {
                                           maxFramesToGrab=atoi(argv[i+1]);
                                           fprintf(stderr,"Setting frame grab to %u \n",maxFramesToGrab);
                                         } else
    if (strcmp(argv[i],"-module")==0)    {
                                           moduleID_1 = getModuleIdFromModuleName(argv[i+1]);
                                           fprintf(stderr,"Overriding Module Used , set to %s ( %u ) \n",getModuleStringName(moduleID_1),moduleID_1);
                                         } else
    if (strcmp(argv[i],"-floodEraseDepthSource")==0)
                                                    {
                                                     segConfDepth.floodErase.pX[segConfDepth.floodErase.totalPoints] = atoi(argv[i+1]);
                                                     segConfDepth.floodErase.pY[segConfDepth.floodErase.totalPoints] = atoi(argv[i+2]);
                                                     segConfDepth.floodErase.threshold[segConfDepth.floodErase.totalPoints] = atoi(argv[i+3]);
                                                     segConfDepth.floodErase.source=1;
                                                     ++segConfDepth.floodErase.totalPoints;
                                                    } else
    if (strcmp(argv[i],"-floodEraseRGBSource")==0)
                                                {
                                                  segConfRGB.floodErase.pX[segConfRGB.floodErase.totalPoints] = atoi(argv[i+1]);
                                                  segConfRGB.floodErase.pY[segConfRGB.floodErase.totalPoints] = atoi(argv[i+2]);
                                                  segConfRGB.floodErase.threshold[segConfRGB.floodErase.totalPoints] = atoi(argv[i+3]);
                                                  segConfRGB.floodErase.source=1;
                                                  ++segConfRGB.floodErase.totalPoints;
                                                } else
    if (strcmp(argv[i],"-floodEraseRGBTarget")==0) {
                                                  segConfRGB.floodErase.pX[segConfRGB.floodErase.totalPoints] = atoi(argv[i+1]);
                                                  segConfRGB.floodErase.pY[segConfRGB.floodErase.totalPoints] = atoi(argv[i+2]);
                                                  segConfRGB.floodErase.threshold[segConfRGB.floodErase.totalPoints] = atoi(argv[i+3]);
                                                  segConfRGB.floodErase.target=1;
                                                  ++segConfRGB.floodErase.totalPoints;
                                                 } else
    if (strcmp(argv[i],"-cropRGB")==0)    { segConfRGB.minX = atoi(argv[i+1]); segConfRGB.minY = atoi(argv[i+2]);
                                            segConfRGB.maxX = atoi(argv[i+3]); segConfRGB.maxY = atoi(argv[i+4]);  doNotSegmentRGB=0; } else
    if (strcmp(argv[i],"-cropDepth")==0)  { segConfDepth.minX = atoi(argv[i+1]); segConfDepth.minY = atoi(argv[i+2]);
                                            segConfDepth.maxX = atoi(argv[i+3]); segConfDepth.maxY = atoi(argv[i+4]);  doNotSegmentDepth=0; } else
    if (strcmp(argv[i],"-minRGB")==0)     { segConfRGB.minR = atoi(argv[i+1]); segConfRGB.minG = atoi(argv[i+2]); segConfRGB.minB = atoi(argv[i+3]); doNotSegmentRGB=0; } else
    if (strcmp(argv[i],"-maxRGB")==0)     { segConfRGB.maxR = atoi(argv[i+1]); segConfRGB.maxG = atoi(argv[i+2]); segConfRGB.maxB = atoi(argv[i+3]); doNotSegmentRGB=0; } else
    if (strcmp(argv[i],"-eraseRGB")==0) { segConfRGB.eraseColorR = atoi(argv[i+1]); segConfRGB.eraseColorG = atoi(argv[i+2]); segConfRGB.eraseColorB = atoi(argv[i+3]);  } else
    if (strcmp(argv[i],"-replaceRGB")==0) { segConfRGB.replaceR = atoi(argv[i+1]); segConfRGB.replaceG = atoi(argv[i+2]); segConfRGB.replaceB = atoi(argv[i+3]); segConfRGB.enableReplacingColors=1; } else
    if (strcmp(argv[i],"-bbox")==0)       {
                                            segConfDepth.enableBBox=1;
                                            segConfDepth.bboxX1=atof(argv[i+1]);
                                            segConfDepth.bboxY1=atof(argv[i+2]);
                                            segConfDepth.bboxZ1=atof(argv[i+3]);
                                            segConfDepth.bboxX2=atof(argv[i+4]);
                                            segConfDepth.bboxY2=atof(argv[i+5]);
                                            segConfDepth.bboxZ2=atof(argv[i+6]);
                                          } else
    if (strcmp(argv[i],"-minDepth")==0)   { segConfDepth.minDepth = atoi(argv[i+1]); doNotSegmentDepth=0; } else
    if (strcmp(argv[i],"-maxDepth")==0)   { segConfDepth.maxDepth = atoi(argv[i+1]); doNotSegmentDepth=0; } else
    if (strcmp(argv[i],"-combine")==0)    { combinationMode=pickCombinationModeFromString(argv[i+1]); }     else
    if (
         (strcmp(argv[i],"-to")==0) ||
         (strcmp(argv[i],"-o")==0)
        )
        {
          strcpy(outputfoldername,"frames/");
          strcat(outputfoldername,argv[i+1]);
          makepath(outputfoldername);
          fprintf(stderr,"OutputPath , set to %s  \n",outputfoldername);
         }
       else
    if (
        (strcmp(argv[i],"-from")==0) ||
        (strcmp(argv[i],"-i")==0)
       )
       { strcat(inputname,argv[i+1]); fprintf(stderr,"Input , set to %s  \n",inputname); }
  }


  if (!acquisitionIsModuleLinked(moduleID_1))
   {
       fprintf(stderr,"The module you are trying to use as module A is not linked in this build of the Acquisition library..\n");
       return 1;
   }

  /*! --------------------------------- INITIALIZATION FROM COMMAND LINE PARAMETERS END --------------------------------- */



  //We need to initialize our module before calling any related calls to the specific module..
  if (!acquisitionStartModule(moduleID_1,16 /*maxDevices*/ , 0 ))
  {
       fprintf(stderr,"Could not start module A %s ..\n",getModuleStringName(moduleID_1));
       return 1;
   }
   //We want to initialize all possible devices in this example..
   unsigned int devID_1=0 ;


   char * devName = inputname;
   if (strlen(inputname)<1) { devName=0; }

    //Initialize Every OpenNI Device
    acquisitionOpenDevice(moduleID_1,devID_1,devName,640,480,25);
    acquisitionMapDepthToRGB(moduleID_1,devID_1);


    usleep(1000); // Waiting a while for the glitch frames to pass
    char outfilename[512]={0};


    unsigned int widthRGB , heightRGB , channelsRGB , bitsperpixelRGB;
    acquisitionGetColorFrameDimensions(moduleID_1,devID_1,&widthRGB,&heightRGB ,&channelsRGB , &bitsperpixelRGB );
    //Todo , check with module 2 bla bla
    char * rgbOut = ( char* )  malloc(widthRGB*heightRGB*channelsRGB * (bitsperpixelRGB/8 ) );

    unsigned int widthDepth , heightDepth , channelsDepth , bitsperpixelDepth;
    acquisitionGetDepthFrameDimensions(moduleID_1,devID_1,&widthDepth,&heightDepth ,&channelsDepth , &bitsperpixelDepth );
    short * depthOut = ( short* )  malloc(widthDepth*heightDepth*channelsDepth * (bitsperpixelDepth/8 ) );



    if (calibrationSet)
    {
      if (!ReadCalibration(calibrationFile,widthDepth,heightDepth,&calib) )
                                             {
                                               fprintf(stderr,"Could not read calibration file `%s`\n",calibrationFile);
                                               return 1;
                                             }
    }


   float centerX;
   float centerY;
   float centerZ;

   for (frameNum=0; frameNum<maxFramesToGrab; frameNum++)
    {
        acquisitionStartTimer(0);

        acquisitionSnapFrames(moduleID_1,devID_1);


        unsigned long colorTimestamp = acquisitionGetColorTimestamp(moduleID_1,devID_1);
        unsigned char * segmentedRGB = copyRGB(acquisitionGetColorFrame(moduleID_1,devID_1) ,widthRGB , heightRGB);

        unsigned long depthTimestamp = acquisitionGetDepthTimestamp(moduleID_1,devID_1);
        unsigned short * segmentedDepth = copyDepth(acquisitionGetDepthFrame(moduleID_1,devID_1) ,widthDepth , heightDepth);

        segmentRGBAndDepthFrame (
                                   segmentedRGB ,
                                   segmentedDepth ,
                                   widthRGB , heightRGB,
                                   &segConfRGB ,
                                   &segConfDepth ,
                                   &calib ,
                                   combinationMode
                                );



        sprintf(outfilename,"%s/colorFrame_%u_%05u.pnm",outputfoldername,devID_1,frameNum);
        acquisitionSimulateTime( colorTimestamp );
        saveRawImageToFile(outfilename,segmentedRGB,widthRGB,heightRGB,channelsRGB,bitsperpixelRGB);


        sprintf(outfilename,"%s/depthFrame_%u_%05u.pnm",outputfoldername,devID_1,frameNum);
        acquisitionSimulateTime( depthTimestamp );
        saveRawImageToFile(outfilename,(char*) segmentedDepth,widthDepth,heightDepth,channelsDepth,bitsperpixelDepth);
        //getDepthBlobAverage(&centerX,&centerY,&centerZ,segmentedDepth,widthDepth,heightDepth);

       free (segmentedRGB);
       free (segmentedDepth);


       acquisitionStopTimer(0);
       if (frameNum%25==0) fprintf(stderr,"%0.2f fps\n",acquisitionGetTimerFPS(0));
    }


    fprintf(stderr,"Done grabbing %u frames! \n",maxFramesToGrab);
    acquisitionCloseDevice(moduleID_1,devID_1);

    acquisitionStopModule(moduleID_1);
    return 0;
}
