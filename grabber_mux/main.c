#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "../acquisition/Acquisition.h"
#include "../acquisition_mux/AcquisitionMux.h"

char outputfoldername[512]={0};

int enforceCalibrationSharing = 1;
char inputname1[512]={0};
char inputname2[512]={0};


unsigned int defaultWidth=640;
unsigned int defaultHeight=480;


int calibrationSetA = 0;
struct calibration calibA;
int calibrationSetB = 0;
struct calibration calibB;


int makepath(char * path)
{
    //FILE *fp;
    /* Open the command for reading. */
    char command[1024];
    sprintf(command,"mkdir -p %s",outputfoldername);
    fprintf(stderr,"Executing .. %s \n",command);

    return system(command);
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
  ModuleIdentifier moduleID_1 = OPENGL_ACQUISITION_MODULE;//OPENNI1_ACQUISITION_MODULE;//
  ModuleIdentifier moduleID_2 = TEMPLATE_ACQUISITION_MODULE;//OPENNI1_ACQUISITION_MODULE;//
  strcpy(outputfoldername,"frames/");

  /*! --------------------------------- INITIALIZATION FROM COMMAND LINE PARAMETERS --------------------------------- */

  int i=0;
  for (i=0; i<argc; i++)
  {
    if (strcmp(argv[i],"-calibration1")==0) {
                                             calibrationSetA=1;
                                             if (!ReadCalibration(argv[i+1],defaultWidth,defaultHeight,&calibA) )
                                             {
                                               fprintf(stderr,"Could not read calibration file for file 1 `%s`\n",argv[i+1]);
                                               return 1;
                                             }
                                           } else
    if (strcmp(argv[i],"-calibration2")==0) {
                                             calibrationSetB=1;
                                             if (!ReadCalibration(argv[i+1],defaultWidth,defaultHeight,&calibB) )
                                             {
                                               fprintf(stderr,"Could not read calibration file for file 2 `%s`\n",argv[i+1]);
                                               return 1;
                                             }
                                           } else
    if (strcmp(argv[i],"-maxFrames")==0) {
                                           maxFramesToGrab=atoi(argv[i+1]);
                                           fprintf(stderr,"Setting frame grab to %u \n",maxFramesToGrab);
                                         } else
    if (strcmp(argv[i],"-module1")==0)    {
                                           moduleID_1 = getModuleIdFromModuleName(argv[i+1]);
                                           fprintf(stderr,"Overriding Module 1 Used , set to %s ( %u ) \n",getModuleStringName(moduleID_1),moduleID_1);
                                         } else
    if (strcmp(argv[i],"-module2")==0)   {
                                           moduleID_2 = getModuleIdFromModuleName(argv[i+1]);
                                           fprintf(stderr,"Overriding Module 2 Used , set to %s ( %u ) \n",getModuleStringName(moduleID_2),moduleID_2);
                                         } else
    if (
        (strcmp(argv[i],"-from1")==0) ||
        (strcmp(argv[i],"-i1")==0)
       )
       { strcat(inputname1,argv[i+1]); fprintf(stderr,"Input , set to %s  \n",inputname1); }
      else
    if (
         (strcmp(argv[i],"-from2")==0)||
         (strcmp(argv[i],"-i2")==0)
       )
       { strcat(inputname2,argv[i+1]); fprintf(stderr,"Input , set to %s  \n",inputname2); }
      else
    if (
        (strcmp(argv[i],"-o")==0) ||
        (strcmp(argv[i],"-to")==0)
       )
                                         {
                                           strcpy(outputfoldername,"frames/");
                                           strcat(outputfoldername,argv[i+1]);
                                           makepath(outputfoldername);
                                           fprintf(stderr,"OutputPath , set to %s  \n",outputfoldername);
                                         }
  }


  if (!acquisitionIsModuleLinked(moduleID_1))
   {
       fprintf(stderr,"The module you are trying to use as module A is not linked in this build of the Acquisition library..\n");
       return 1;
   }

  if (!acquisitionIsModuleLinked(moduleID_2))
   {
       fprintf(stderr,"The module you are trying to use as module B is not linked in this build of the Acquisition library..\n");
       return 1;
   }
  /*! --------------------------------- INITIALIZATION FROM COMMAND LINE PARAMETERS END --------------------------------- */





  //We need to initialize our module before calling any related calls to the specific module..
  if (!acquisitionStartModule(moduleID_1,16 /*maxDevices*/ , 0 ))
  {
       fprintf(stderr,"Could not start module A %s ..\n",getModuleStringName(moduleID_1));
       return 1;
   }
  if (!acquisitionStartModule(moduleID_2,16 /*maxDevices*/ , 0 ))
  {
       fprintf(stderr,"Could not start module B %s ..\n",getModuleStringName(moduleID_2));
       return 1;
   }


  //We want to initialize all possible devices in this example..
  unsigned int devID_1=0 , devID_2=0;
  if (moduleID_1==moduleID_2) { ++devID_2; }

   if ( calibrationSetA )
   {
    fprintf(stderr,"Set Far/Near to %f/%f\n",calibA.farPlane,calibA.nearPlane);
    acquisitionSetColorCalibration(moduleID_1,devID_1,&calibA);
    acquisitionSetDepthCalibration(moduleID_1,devID_1,&calibA);
   }

   if ( calibrationSetB )
   {
    fprintf(stderr,"Set Far/Near to %f/%f\n",calibB.farPlane,calibB.nearPlane);
    acquisitionSetColorCalibration(moduleID_2,devID_2,&calibB);
    acquisitionSetDepthCalibration(moduleID_2,devID_2,&calibB);
   }


  char * devName1 = inputname1;
  if (strlen(inputname1)<1) { devName1=0; }
  char * devName2 = inputname2;
  if (strlen(inputname2)<1) { devName2=0; }

    //Initialize Every OpenNI Device
    acquisitionOpenDevice(moduleID_1,devID_1,devName1,defaultWidth,defaultHeight,25);
    acquisitionMapDepthToRGB(moduleID_1,devID_1);

    acquisitionOpenDevice(moduleID_2,devID_2,devName2,defaultWidth,defaultHeight,25);
    acquisitionMapDepthToRGB(moduleID_2,devID_2);


    if ( enforceCalibrationSharing )
    {
      struct calibration calib={0};
      acquisitionGetColorCalibration(moduleID_1,devID_1,&calib);
      acquisitionSetColorCalibration(moduleID_2,devID_2,&calib);
      acquisitionGetDepthCalibration(moduleID_1,devID_1,&calib);
      acquisitionSetDepthCalibration(moduleID_2,devID_2,&calib);
    }

    usleep(1000); // Waiting a while for the glitch frames to pass
    char outfilename[512]={0};


    unsigned int widthRGB , heightRGB , channelsRGB , bitsperpixelRGB;
    acquisitionGetColorFrameDimensions(moduleID_1,devID_1,&widthRGB,&heightRGB ,&channelsRGB , &bitsperpixelRGB );
    //Todo , check with module 2 bla bla
    char * rgbOut = ( char* )  malloc(widthRGB*heightRGB*channelsRGB * (bitsperpixelRGB/8 ) );

    unsigned int widthDepth , heightDepth , channelsDepth , bitsperpixelDepth;
    acquisitionGetDepthFrameDimensions(moduleID_1,devID_1,&widthDepth,&heightDepth ,&channelsDepth , &bitsperpixelDepth );
    short * depthOut = ( short* )  malloc(widthDepth*heightDepth*channelsDepth * (bitsperpixelDepth/8 ) );


    fprintf(stderr,"Base Module is %s , device %u , Overlay Module is %s , device %u\n",getModuleStringName(moduleID_1),devID_1, getModuleStringName(moduleID_2),devID_2);

   for (frameNum=0; frameNum<maxFramesToGrab; frameNum++)
    {

        acquisitionSnapFrames(moduleID_1,devID_1);
        acquisitionSnapFrames(moduleID_2,devID_2);


        mux2RGBAndDepthFrames
          (
           acquisitionGetColorFrame(moduleID_1,devID_1) , //Module 1 is Base
           acquisitionGetColorFrame(moduleID_2,devID_2) , //Module 2 is Overlay
           rgbOut ,
           acquisitionGetDepthFrame(moduleID_1,devID_1) ,
           acquisitionGetDepthFrame(moduleID_2,devID_2) ,
           depthOut ,
           widthRGB , heightRGB , 0 );


        sprintf(outfilename,"%s/colorFrame_%u_%05u",outputfoldername,devID_1,frameNum);
        saveMuxImageToFile(outfilename,rgbOut,widthRGB , heightRGB, channelsRGB , bitsperpixelRGB);

        sprintf(outfilename,"%s/depthFrame_%u_%05u",outputfoldername,devID_1,frameNum);
        saveMuxImageToFile(outfilename,(char*) depthOut,widthDepth , heightDepth, channelsDepth , bitsperpixelDepth);


        sprintf(outfilename,"%s/BASEcolorFrame_%u_%05u",outputfoldername,devID_1,frameNum);
        acquisitionSaveColorFrame(moduleID_1,devID_1,outfilename);
        sprintf(outfilename,"%s/BASEdepthFrame_%u_%05u",outputfoldername,devID_1,frameNum);
        acquisitionSaveDepthFrame(moduleID_1,devID_1,outfilename);

        sprintf(outfilename,"%s/OVERLAYcolorFrame_%u_%05u",outputfoldername,devID_2,frameNum);
        acquisitionSaveColorFrame(moduleID_2,devID_2,outfilename);
        sprintf(outfilename,"%s/OVERLAYdepthFrame_%u_%05u",outputfoldername,devID_2,frameNum);
        acquisitionSaveDepthFrame(moduleID_2,devID_2,outfilename);

    }




    fprintf(stderr,"Done grabbing %u frames! \n",maxFramesToGrab);
    acquisitionCloseDevice(moduleID_1,devID_1);
    acquisitionCloseDevice(moduleID_2,devID_2);


    acquisitionStopModule(moduleID_1);
    acquisitionStopModule(moduleID_2);


    return 0;
}


