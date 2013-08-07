#include "Acquisition.h"
#include "acquisition_setup.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>



#if USE_V4L2
#include "../v4l2_acquisition_shared_library/V4L2Acquisition.h"
#include "../v4l2stereo_acquisition_shared_library/V4L2StereoAcquisition.h"
#endif

#if USE_OPENNI1
#include "../openni1_acquisition_shared_library/OpenNI1Acquisition.h"
#endif

#if USE_OPENNI2
#include "../openni2_acquisition_shared_library/OpenNI2Acquisition.h"
#endif

#if USE_FREENECT
#include "../libfreenect_acquisition_shared_library/FreenectAcquisition.h"
#endif


#if USE_OPENGL
#include "../opengl_acquisition_shared_library/OpenGLAcquisition.h"
//#include "../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/OGLRendererSandbox.h"
#endif


#if USE_TEMPLATE
#include "../template_acquisition_shared_library/TemplateAcquisition.h"
#endif


#define EPOCH_YEAR_IN_TM_YEAR 1900

//This should probably pass on to each of the sub modules
float minDistance = -10;
float scaleFactor = 0.0021;

const char *days[] = {"Sun","Mon","Tue","Wed","Thu","Fri","Sat"};
const char *months[] = {"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"};

unsigned long tickBase = 0;

unsigned int simulateTick=0;
unsigned long simulatedTickValue=0;

int acquisitionSimulateTime(unsigned long timeInMillisecs)
{
  simulateTick=1;
  simulatedTickValue=timeInMillisecs;
  return 1;
}


unsigned long GetTickCount()
{
   if (simulateTick) { return simulatedTickValue; }

   //This returns a monotnic "uptime" value in milliseconds , it behaves like windows GetTickCount() but its not the same..
   struct timespec ts;
   if ( clock_gettime(CLOCK_MONOTONIC,&ts) != 0) { fprintf(stderr,"Error Getting Tick Count\n"); return 0; }

   if (tickBase==0)
   {
     tickBase = ts.tv_sec*1000 + ts.tv_nsec/1000000;
     return 0;
   }

   return ( ts.tv_sec*1000 + ts.tv_nsec/1000000 ) - tickBase;
}

/*
int GetDateString(char * output,char * label,unsigned int now,unsigned int dayofweek,unsigned int day,unsigned int month,unsigned int year,unsigned int hour,unsigned int minute,unsigned int second)
{
   //Date: Sat, 29 May 2010 12:31:35 GMT
   //Last-Modified: Sat, 29 May 2010 12:31:35 GMT
   if ( now )
      {
        time_t clock = time(NULL);
        struct tm * ptm = gmtime ( &clock );

        sprintf(output,"%s: %s, %u %s %u %02u:%02u:%02u GMT\n",label,days[ptm->tm_wday],ptm->tm_mday,months[ptm->tm_mon],EPOCH_YEAR_IN_TM_YEAR+ptm->tm_year,ptm->tm_hour,ptm->tm_min,ptm->tm_sec);

      } else
      {
        sprintf(output,"%s: %s, %u %s %u %02u:%02u:%02u GMT\n",label,days[dayofweek],day,months[month],year,hour,minute,second);
      }
    return 1;
}*/




unsigned int simplePow(unsigned int base,unsigned int exp)
{
    if (exp==0) return 1;
    unsigned int retres=base;
    unsigned int i=0;
    for (i=0; i<exp-1; i++)
    {
        retres*=base;
    }
    return retres;
}



int savePCD_PointCloud(char * filename , short * depthFrame , char * colorFrame , unsigned int width , unsigned int height , float cx , float cy , float fx , float fy )
{
    if(depthFrame==0) { fprintf(stderr,"saveToPCD_PointCloud(%s) called for an unallocated (empty) depth frame , will not write any file output\n",filename); return 0; }
    if(colorFrame==0) { fprintf(stderr,"saveToPCD_PointCloud(%s) called for an unallocated (empty) color frame , will not write any file output\n",filename); return 0; }

    FILE *fd=0;
    fd = fopen(filename,"wb");
    if (fd!=0)
    {
        fprintf(fd, "# .PCD v.5 - Point Cloud Data file format\n");
        fprintf(fd, "FIELDS x y z rgb\n");
        fprintf(fd, "SIZE 4 4 4 4\n");
        fprintf(fd, "TYPE F F F U\n");
        fprintf(fd, "WIDTH %u\n",width);
        fprintf(fd, "HEIGHT %u\n",height);
        fprintf(fd, "POINTS %u\n",height*width);
        fprintf(fd, "DATA ascii\n");

        short * depthPTR = depthFrame;
        char  * colorPTR = colorFrame;

        unsigned int px=0,py=0;
        float x=0.0,y=0.0,z=0.0;
        unsigned char * r , * b , * g;
        unsigned int rgb=0;

        for (py=0; py<height; py++)
        {
         for (px=0; px<width; px++)
         {
           z = * depthPTR; ++depthPTR;
           x = (px - cx) * (z + minDistance) * scaleFactor * (width/height) ;
           y = (py - cy) * (z + minDistance) * scaleFactor;

           r=colorPTR; ++colorPTR;
           g=colorPTR; ++colorPTR;
           b=colorPTR; ++colorPTR;


        /* To pack it :
            int rgb = ((int)r) << 16 | ((int)g) << 8 | ((int)b);

           To unpack it :
            int rgb = ...;
            uint8_t r = (rgb >> 16) & 0x0000ff;
            uint8_t g = (rgb >> 8) & 0x0000ff;
            uint8_t b = (rgb) & 0x0000ff; */
           rgb = ((int)*r) << 16 | ((int)*g) << 8 | ((int)*b);

           fprintf(fd, "%0.4f %0.4f %0.4f %u\n",x,y,z,rgb);
         }
        }
        fclose(fd);
        return 1;
    }
    else
    {
        fprintf(stderr,"SaveRawImageToFile could not open output file %s\n",filename);
        return 0;
    }

   return 0;
}




int saveRawImageToFile(char * filename,char * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel)
{
    //fprintf(stderr,"saveRawImageToFile(%s) called\n",filename);

    if ( (width==0) || (height==0) || (channels==0) || (bitsperpixel==0) ) { fprintf(stderr,"saveRawImageToFile(%s) called with zero dimensions\n",filename); return 0;}
    if(pixels==0) { fprintf(stderr,"saveRawImageToFile(%s) called for an unallocated (empty) frame , will not write any file output\n",filename); return 0; }
    if (bitsperpixel>16) { fprintf(stderr,"PNM does not support more than 2 bytes per pixel..!\n"); return 0; }

    FILE *fd=0;
    fd = fopen(filename,"wb");

    if (fd!=0)
    {
        unsigned int n;
        if (channels==3) fprintf(fd, "P6\n");
        else if (channels==1) fprintf(fd, "P5\n");
        else
        {
            fprintf(stderr,"Invalid channels arg (%u) for SaveRawImageToFile\n",channels);
            fclose(fd);
            return 1;
        }

        char output[256]={0};
      /*GetDateString(output,"TIMESTAMP",1,0,0,0,0,0,0,0);
        fprintf(fd, "#%s\n", output );*/

        fprintf(fd, "#TIMESTAMP %u\n",GetTickCount());


        fprintf(fd, "%d %d\n%u\n", width, height , simplePow(2 ,bitsperpixel)-1);

        float tmp_n = (float) bitsperpixel/ 8;
        tmp_n = tmp_n *  width * height * channels ;
        n = (unsigned int) tmp_n;

        fwrite(pixels, 1 , n , fd);
        fflush(fd);
        fclose(fd);
        return 1;
    }
    else
    {
        fprintf(stderr,"SaveRawImageToFile could not open output file %s\n",filename);
        return 0;
    }
    return 0;
}

//Ok this is basically casting the 2 bytes of depth into 3 RGB bytes leaving one color channel off (the blue one)
//depth is casted to char to simplify things , but that adds sizeof(short) to the pointer arethemetic!
char * convertShortDepthToRGBDepth(short * depth,unsigned int width , unsigned int height)
{
  if (depth==0)  { fprintf(stderr,"Depth is not allocated , cannot perform DepthToRGB transformation \n"); return 0; }
  char * depthPTR= (char*) depth; // This will be the traversing pointer for input
  char * depthLimit = (char*) depth + width*height * sizeof(short); //<- we use sizeof(short) because we have casted to char !


  char * outFrame = (char*) malloc(width*height*3*sizeof(char));
  if (outFrame==0) { fprintf(stderr,"Could not perform DepthToRGB transformation\nNo memory for new frame\n"); return 0; }

  char * outFramePTR = outFrame; // This will be the traversing pointer for output
  while ( depthPTR<depthLimit )
  {
     * outFramePTR = *depthPTR; ++outFramePTR; ++depthPTR;
     * outFramePTR = *depthPTR; ++outFramePTR; ++depthPTR;
     * outFramePTR = 0;         ++outFramePTR;
  }
 return outFrame;
}


//Ok this is basically casting the 2 bytes of depth into 3 RGB bytes leaving one color channel off (the blue one)
//depth is casted to char to simplify things , but that adds sizeof(short) to the pointer arethemetic!
char * convertShortDepthToCharDepth(short * depth,unsigned int width , unsigned int height , unsigned int min_depth , unsigned int max_depth)
{
  if (depth==0)  { fprintf(stderr,"Depth is not allocated , cannot perform DepthToRGB transformation \n"); return 0; }
  short * depthPTR= depth; // This will be the traversing pointer for input
  short * depthLimit =  depth + width*height; //<- we use sizeof(short) because we have casted to char !


  char * outFrame = (char*) malloc(width*height*1*sizeof(char));
  if (outFrame==0) { fprintf(stderr,"Could not perform DepthToRGB transformation\nNo memory for new frame\n"); return 0; }

  float depth_range = max_depth-min_depth;
  if (depth_range ==0 ) { depth_range = 1; }
  float multiplier = 255 / depth_range;


  char * outFramePTR = outFrame; // This will be the traversing pointer for output
  while ( depthPTR<depthLimit )
  {
     unsigned int scaled = (unsigned int) (*depthPTR) * multiplier;
     unsigned char scaledChar = (unsigned char) scaled;
     * outFramePTR = scaledChar;

     ++outFramePTR;
     ++depthPTR;
  }
 return outFrame;
}



int acquisitionGetModulesCount()
{
  unsigned int modules_linked = 0;


  #if USE_V4L2
   modules_linked+=2;
   fprintf(stderr,"V4L2 code linked\n");
   fprintf(stderr,"V4L2Stereo code linked\n");
  #endif

  #if USE_OPENGL
   ++modules_linked;
   fprintf(stderr,"OpenGL code linked\n");
  #endif

  #if USE_OPENNI1
   ++modules_linked;
   fprintf(stderr,"OpenNI1 code linked\n");
  #endif

  #if USE_OPENNI2
   ++modules_linked;
   fprintf(stderr,"OpenNI2 code linked\n");
  #endif

  #if USE_FREENECT
   ++modules_linked;
   fprintf(stderr,"Freenect code linked\n");
  #endif

  #if USE_TEMPLATE
   ++modules_linked;
   fprintf(stderr,"Template code linked\n");
  #endif

  return modules_linked;

}


ModuleIdentifier getModuleIdFromModuleName(char * moduleName)
{
   ModuleIdentifier moduleID = 0;
          if (strcasecmp("FREENECT",moduleName)==0 )  { moduleID = FREENECT_ACQUISITION_MODULE; } else
          if (strcasecmp("OPENNI",moduleName)==0 )  { moduleID = OPENNI1_ACQUISITION_MODULE;  } else
          if (strcasecmp("OPENNI1",moduleName)==0 )  { moduleID = OPENNI1_ACQUISITION_MODULE;  } else
          if (strcasecmp("OPENNI2",moduleName)==0 )  { moduleID = OPENNI2_ACQUISITION_MODULE;  } else
          if (strcasecmp("OPENGL",moduleName)==0 )   { moduleID = OPENGL_ACQUISITION_MODULE;   } else
          if (strcasecmp("V4L2",moduleName)==0 )   { moduleID = V4L2_ACQUISITION_MODULE;   } else
          if (strcasecmp("V4L2STEREO",moduleName)==0 )   { moduleID = V4L2STEREO_ACQUISITION_MODULE;   } else
          if (strcasecmp("TEMPLATE",moduleName)==0 )  { moduleID = TEMPLATE_ACQUISITION_MODULE; }
   return moduleID;
}


char * getModuleStringName(ModuleIdentifier moduleID)
{
  switch (moduleID)
    {
      case V4L2_ACQUISITION_MODULE    :  return (char*) "V4L2 MODULE"; break;
      case V4L2STEREO_ACQUISITION_MODULE    :  return (char*) "V4L2STEREO MODULE"; break;
      case FREENECT_ACQUISITION_MODULE:  return (char*) "FREENECT MODULE"; break;
      case OPENNI1_ACQUISITION_MODULE :  return (char*) "OPENNI1 MODULE"; break;
      case OPENNI2_ACQUISITION_MODULE :  return (char*) "OPENNI2 MODULE"; break;
      case OPENGL_ACQUISITION_MODULE :  return (char*) "OPENGL MODULE"; break;
      case TEMPLATE_ACQUISITION_MODULE    :  return (char*) "TEMPLATE MODULE"; break;
    };
    return (char*) "UNKNOWN MODULE";
}



int acquisitionIsModuleLinked(ModuleIdentifier moduleID)
{
    switch (moduleID)
    {
      #if USE_V4L2
      case V4L2_ACQUISITION_MODULE    :
          return 1;
      break;
      case V4L2STEREO_ACQUISITION_MODULE    :
          return 1;
      break;
      #endif
      case OPENGL_ACQUISITION_MODULE    :
        #if USE_OPENGL
          return 1;
        #endif
      break;
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          return 1;
        #endif
      break;
      case FREENECT_ACQUISITION_MODULE:
        #if USE_FREENECT
          return 1;
        #endif
      break;
      case OPENNI1_ACQUISITION_MODULE :
        #if USE_OPENNI1
          return 1;
        #endif
      break;
      case OPENNI2_ACQUISITION_MODULE :
        #if USE_OPENNI2
          return 1;
        #endif
      break;
    };

    return 0;
}

void MeaningfullWarningMessage(ModuleIdentifier moduleFailed,DeviceIdentifier devFailed,char * fromFunction)
{
  if (!acquisitionIsModuleLinked(moduleFailed))
   {
       fprintf(stderr,"%s is not linked in this build of the acquisition library system..\n",getModuleStringName(moduleFailed));
       return ;
   }

   fprintf(stderr,"%s hasn't got an implementation for function %s ..\n",getModuleStringName(moduleFailed),fromFunction);

}


/*! ------------------------------------------
    BASIC START STOP MECHANISMS FOR MODULES..
   ------------------------------------------*/
int acquisitionStartModule(ModuleIdentifier moduleID,unsigned int maxDevices,char * settings)
{
    switch (moduleID)
    {
      #if USE_V4L2
      case V4L2_ACQUISITION_MODULE    :
          return startV4L2(maxDevices,settings);
      break;
      case V4L2STEREO_ACQUISITION_MODULE    :
          return startV4L2Stereo(maxDevices,settings);
      break;
      #endif
      case OPENGL_ACQUISITION_MODULE    :
        #if USE_OPENGL
          return 1;
        #endif
      break;
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          return startTemplate(maxDevices,settings);
        #endif
      break;
      case FREENECT_ACQUISITION_MODULE:
        #if USE_FREENECT
          return startFreenectModule(maxDevices);
        #endif
      break;
      case OPENNI1_ACQUISITION_MODULE :
        #if USE_OPENNI1
          return startOpenNI1(maxDevices);
        #endif
      break;
      case OPENNI2_ACQUISITION_MODULE :
        #if USE_OPENNI2
          return startOpenNI2(maxDevices);
        #endif
      break;
    };

    MeaningfullWarningMessage(moduleID,0,"acquisitionStartModule");
    return 0;
}


int acquisitionStopModule(ModuleIdentifier moduleID)
{
    switch (moduleID)
    {
      #if USE_V4L2
      case V4L2_ACQUISITION_MODULE    :
          return stopV4L2();
      break;
      case V4L2STEREO_ACQUISITION_MODULE    :
          return stopV4L2Stereo();
      break;
      #endif
      case OPENGL_ACQUISITION_MODULE    :
        #if USE_OPENGL
          return 1;
        #endif
      break;
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          return stopTemplate();
        #endif
      break;
      case FREENECT_ACQUISITION_MODULE: break;
      case OPENNI1_ACQUISITION_MODULE :
        #if USE_OPENNI1
          return stopOpenNI1();
        #endif
      break;
      case OPENNI2_ACQUISITION_MODULE :
        #if USE_OPENNI2
          return stopOpenNI2();
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,0,"acquisitionStopModule");
    return 0;
}


int acquisitionGetModuleDevices(ModuleIdentifier moduleID)
{
    switch (moduleID)
    {
      #if USE_V4L2
      case V4L2_ACQUISITION_MODULE    :
          return getV4L2NumberOfDevices();
      break;
      case V4L2STEREO_ACQUISITION_MODULE    :
          return getV4L2StereoNumberOfDevices();
      break;
      #endif
      case OPENGL_ACQUISITION_MODULE    :
        #if USE_OPENGL
          return getOpenGLNumberOfDevices();
        #endif
      break;
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          return getTemplateNumberOfDevices();
        #endif
      break;
      case FREENECT_ACQUISITION_MODULE:
        #if USE_FREENECT
         return getFreenectNumberOfDevices();
        #endif
      break;
      case OPENNI1_ACQUISITION_MODULE :
        #if USE_OPENNI1
          return getOpenNI1NumberOfDevices();
        #endif
      break;
      case OPENNI2_ACQUISITION_MODULE :
        #if USE_OPENNI2
          return getOpenNI2NumberOfDevices();
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,0,"acquisitionGetModuleDevices");
    return 0;
}



/*! ------------------------------------------
    FRAME SNAPPING MECHANISMS FOR MODULES..
   ------------------------------------------*/
int acquisitionOpenDevice(ModuleIdentifier moduleID,DeviceIdentifier devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate)
{
    switch (moduleID)
    {
      #if USE_V4L2
      case V4L2_ACQUISITION_MODULE    :
           return createV4L2Device(devID,devName,width,height,framerate);
      break;
      case V4L2STEREO_ACQUISITION_MODULE    :
           return createV4L2StereoDevice(devID,devName,width,height,framerate);
      break;
      #endif // USE_V4L2
      case OPENGL_ACQUISITION_MODULE    :
        #if USE_OPENGL
          return createOpenGLDevice(devID,devName,width,height,framerate);
        #endif
      break;
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          return createTemplateDevice(devID,devName,width,height,framerate);
        #endif
      break;
      case FREENECT_ACQUISITION_MODULE:   break;
      case OPENNI1_ACQUISITION_MODULE :
        #if USE_OPENNI1
          return createOpenNI1Device(devID,width,height,framerate);
        #endif
      break;
      case OPENNI2_ACQUISITION_MODULE :
        #if USE_OPENNI2
          return createOpenNI2Device(devID,width,height,framerate);
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,devID,"acquisitionOpenDevice");
    return 0;
}

 int acquisitionCloseDevice(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
    switch (moduleID)
    {
      #if USE_V4L2
      case V4L2_ACQUISITION_MODULE    :
          return destroyV4L2Device(devID);
      break;
      case V4L2STEREO_ACQUISITION_MODULE    :
          return destroyV4L2StereoDevice(devID);
      break;
      #endif
      case OPENGL_ACQUISITION_MODULE    :
        #if USE_OPENGL
          return destroyOpenGLDevice(devID);
        #endif
      break;
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          return destroyTemplateDevice(devID);
        #endif
      break;
      case FREENECT_ACQUISITION_MODULE:   break;
      case OPENNI1_ACQUISITION_MODULE :
        #if USE_OPENNI1
          return destroyOpenNI1Device(devID);
        #endif
      break;
      case OPENNI2_ACQUISITION_MODULE :
        #if USE_OPENNI2
          return destroyOpenNI2Device(devID);
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,devID,"acquisitionCloseDevice");
    return 0;
}


 int acquisitionSeekFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int seekFrame)
{
    switch (moduleID)
    {
      case OPENGL_ACQUISITION_MODULE    :  break;
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          return seekTemplateFrame(devID,seekFrame);
        #endif
      break;
      case FREENECT_ACQUISITION_MODULE:   break;
      case OPENNI1_ACQUISITION_MODULE :   break;
      case OPENNI2_ACQUISITION_MODULE : break;
    };
    MeaningfullWarningMessage(moduleID,devID,"acquisitionSeekFrame");
    return 0;
}


 int acquisitionSnapFrames(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
    //fprintf(stderr,"acquisitionSnapFrames called moduleID=%u devID=%u\n",moduleID,devID);
    switch (moduleID)
    {
      #if USE_V4L2
      case V4L2_ACQUISITION_MODULE    :
          return snapV4L2Frames(devID);
      break;
      case V4L2STEREO_ACQUISITION_MODULE    :
          return snapV4L2StereoFrames(devID);
      break;
      #endif
      case OPENGL_ACQUISITION_MODULE    :
        #if USE_OPENGL
          return snapOpenGLFrames(devID) ;
        #endif
      break;
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          return snapTemplateFrames(devID);
        #endif
      break;
      case FREENECT_ACQUISITION_MODULE:   break;
      case OPENNI1_ACQUISITION_MODULE :
        #if USE_OPENNI1
          return snapOpenNI1Frames(devID);
        #endif
      break;
      case OPENNI2_ACQUISITION_MODULE :
        #if USE_OPENNI2
          return snapOpenNI2Frames(devID);
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,devID,"acquisitionSnapFrames");
    return 0;
}

 int acquisitionSaveColorFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename)
{
    char filenameFull[2048]={0};
    sprintf(filenameFull,"%s.pnm",filename);

    switch (moduleID)
    {
      #if USE_V4L2
      case V4L2_ACQUISITION_MODULE    :
          return saveRawImageToFile(filename,getV4L2ColorPixels(devID),getV4L2ColorWidth(devID),getV4L2ColorHeight(devID),getV4L2ColorChannels(devID),getV4L2ColorBitsPerPixel(devID));
      break;
      case V4L2STEREO_ACQUISITION_MODULE    :
         sprintf(filenameFull,"%s_0.pnm",filename);
         saveRawImageToFile(filenameFull,getV4L2StereoColorPixelsLeft(devID),getV4L2StereoColorWidth(devID),getV4L2StereoColorHeight(devID),getV4L2StereoColorChannels(devID),getV4L2StereoColorBitsPerPixel(devID));
         sprintf(filenameFull,"%s_1.pnm",filename);
         saveRawImageToFile(filenameFull,getV4L2StereoColorPixelsRight(devID),getV4L2StereoColorWidth(devID),getV4L2StereoColorHeight(devID),getV4L2StereoColorChannels(devID),getV4L2StereoColorBitsPerPixel(devID));
        return 1;
      break;
      #endif
      case OPENGL_ACQUISITION_MODULE    :
        #if USE_OPENGL
          return saveRawImageToFile(filenameFull,getOpenGLColorPixels(devID),getOpenGLColorWidth(devID),getOpenGLColorHeight(devID),getOpenGLColorChannels(devID),getOpenGLColorBitsPerPixel(devID));
        #endif
      break;
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          return saveRawImageToFile(filenameFull,getTemplateColorPixels(devID),getTemplateColorWidth(devID),getTemplateColorHeight(devID) ,getTemplateColorChannels(devID),getTemplateColorBitsPerPixel(devID));
        #endif
      break;
      case FREENECT_ACQUISITION_MODULE:
        #if USE_FREENECT
          return saveRawImageToFile(filenameFull,getFreenectColorPixels(devID),getFreenectColorWidth(devID),getFreenectColorHeight(devID) ,getFreenectColorChannels(devID),getFreenectColorBitsPerPixel(devID));
        #endif
      break;
      case OPENNI1_ACQUISITION_MODULE :
        #if USE_OPENNI1
          return saveRawImageToFile(filenameFull,getOpenNI1ColorPixels(devID),getOpenNI1ColorWidth(devID),getOpenNI1ColorHeight(devID) ,getOpenNI1ColorChannels(devID),getOpenNI1ColorBitsPerPixel(devID));
        #endif
      break;
      case OPENNI2_ACQUISITION_MODULE :
        #if USE_OPENNI2
          return saveRawImageToFile(filenameFull,getOpenNI2ColorPixels(devID),getOpenNI2ColorWidth(devID),getOpenNI2ColorHeight(devID) ,getOpenNI2ColorChannels(devID),getOpenNI2ColorBitsPerPixel(devID));
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,devID,"acquisitionSaveColorFrame");
    return 0;
}



 int acquisitionSavePCDPointCoud(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename)
{
  unsigned int width;
  unsigned int height;
  unsigned int channels;
  unsigned int bitsperpixel;
  acquisitionGetDepthFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);

  return savePCD_PointCloud(filename,acquisitionGetDepthFrame(moduleID,devID),acquisitionGetColorFrame(moduleID,devID),
                            width,height,width/2,height/2, 1.0 /*DUMMY fx*/, 1.0 /*DUMMY fy*/ );
}



 int acquisitionSaveDepthFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename)
{
    char filenameFull[2048]={0};
    sprintf(filenameFull,"%s.pnm",filename);

    switch (moduleID)
    {
      #if USE_V4L2
      case V4L2_ACQUISITION_MODULE    :
           fprintf(stderr,"V4L2 Does not have a depth frame\n");
      break;
      case V4L2STEREO_ACQUISITION_MODULE    :
          return saveRawImageToFile(filenameFull,getV4L2StereoDepthPixels(devID),getV4L2StereoDepthWidth(devID),getV4L2StereoDepthHeight(devID),getV4L2StereoDepthChannels(devID),getV4L2StereoDepthBitsPerPixel(devID));
      break;
      #endif
      case OPENGL_ACQUISITION_MODULE    :
        #if USE_OPENGL
          return saveRawImageToFile(filenameFull,(char*) getOpenGLDepthPixels(devID),getOpenGLDepthWidth(devID),getOpenGLDepthHeight(devID),getOpenGLDepthChannels(devID),getOpenGLDepthBitsPerPixel(devID));
        #endif
      break;
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          return saveRawImageToFile(filenameFull,(char*) getTemplateDepthPixels(devID),getTemplateDepthWidth(devID),getTemplateDepthHeight(devID) ,getTemplateDepthChannels(devID),getTemplateDepthBitsPerPixel(devID));
        #endif
      break;
      case FREENECT_ACQUISITION_MODULE:
        #if USE_FREENECT
          return saveRawImageToFile(filenameFull,(char*) getFreenectDepthPixels(devID),getFreenectDepthWidth(devID),getFreenectDepthHeight(devID) ,getFreenectDepthChannels(devID),getFreenectDepthBitsPerPixel(devID));
        #endif
      break;
      case OPENNI1_ACQUISITION_MODULE :
        #if USE_OPENNI1
            return saveRawImageToFile(filenameFull,(char*) getOpenNI1DepthPixels(devID),getOpenNI1DepthWidth(devID),getOpenNI1DepthHeight(devID) ,getOpenNI1DepthChannels(devID),getOpenNI1DepthBitsPerPixel(devID));
        #endif
      break;
      case OPENNI2_ACQUISITION_MODULE :
        #if USE_OPENNI2
            return saveRawImageToFile(filenameFull,(char*) getOpenNI2DepthPixels(devID),getOpenNI2DepthWidth(devID),getOpenNI2DepthHeight(devID) ,getOpenNI2DepthChannels(devID),getOpenNI2DepthBitsPerPixel(devID));
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,devID,"acquisitionSaveDepthFrame");
    return 0;
}

 int acquisitionSaveColoredDepthFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename)
{

    char filenameFull[1024]={0};
    sprintf(filenameFull,"%s.pnm",filename);

    unsigned int width = 0 ;
    unsigned int height = 0 ;
    unsigned int channels = 0 ;
    unsigned int bitsperpixel = 0 ;
    short * inFrame = 0;
    char * outFrame = 0 ;

    inFrame = acquisitionGetDepthFrame(moduleID,devID);
    if (inFrame!=0)
      {
       acquisitionGetDepthFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);
       outFrame = convertShortDepthToRGBDepth(inFrame,width,height);
       if (outFrame!=0)
        {
         saveRawImageToFile(filenameFull,outFrame,width,height,3,8);
         free(outFrame);
         return 1;
        }
      }

    MeaningfullWarningMessage(moduleID,devID,"acquisitionSaveColoredDepthFrame");
    return 0;
}


int acquisitionSaveDepthFrame1C(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename)
{

    char filenameFull[1024]={0};
    sprintf(filenameFull,"%s.pnm",filename);

    unsigned int width = 0 ;
    unsigned int height = 0 ;
    unsigned int channels = 0 ;
    unsigned int bitsperpixel = 0 ;
    short * inFrame = 0;
    char * outFrame = 0 ;

    inFrame = acquisitionGetDepthFrame(moduleID,devID);
    if (inFrame!=0)
      {
       acquisitionGetDepthFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);
       outFrame = convertShortDepthToCharDepth(inFrame,width,height,0,7000);
       if (outFrame!=0)
        {
         saveRawImageToFile(filenameFull,outFrame,width,height,1,8);
         free(outFrame);
         return 1;
        }
      }

    MeaningfullWarningMessage(moduleID,devID,"acquisitionSaveColoredDepthFrame");
    return 0;
}



int acquisitionGetColorCalibration(ModuleIdentifier moduleID,DeviceIdentifier devID,struct calibration * calib)
{
   switch (moduleID)
    {
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          return getTemplateColorCalibration(devID,calib);
        #endif
      break;
      case OPENGL_ACQUISITION_MODULE:
        #if USE_OPENGL
          return getOpenGLColorCalibration(devID,calib);
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,devID,"acquisitionGetColorCalibration");
    return 0;
}

int acquisitionGetDepthCalibration(ModuleIdentifier moduleID,DeviceIdentifier devID,struct calibration * calib)
{
   switch (moduleID)
    {
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          return getTemplateDepthCalibration(devID,calib);
        #endif
      break;
      case OPENGL_ACQUISITION_MODULE:
        #if USE_OPENGL
          return getOpenGLDepthCalibration(devID,calib);
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,devID,"acquisitionGetDepthCalibration");
    return 0;
}





int acquisitionSetColorCalibration(ModuleIdentifier moduleID,DeviceIdentifier devID,struct calibration * calib)
{
   switch (moduleID)
    {
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          return setTemplateColorCalibration(devID,calib);
        #endif
      break;
      case OPENGL_ACQUISITION_MODULE:
        #if USE_OPENGL
          return setOpenGLColorCalibration(devID,calib);
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,devID,"acquisitionGetColorCalibration");
    return 0;
}

int acquisitionSetDepthCalibration(ModuleIdentifier moduleID,DeviceIdentifier devID,struct calibration * calib)
{
   switch (moduleID)
    {
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          return setTemplateDepthCalibration(devID,calib);
        #endif
      break;
      case OPENGL_ACQUISITION_MODULE:
        #if USE_OPENGL
          return setOpenGLDepthCalibration(devID,calib);
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,devID,"acquisitionGetDepthCalibration");
    return 0;
}


unsigned long acquisitionGetColorTimestamp(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
   switch (moduleID)
    {
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          return getLastTemplateColorTimestamp(devID);
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,devID,"acquisitionGetColorTimestamp");
    return 0;
}

unsigned long acquisitionGetDepthTimestamp(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
   switch (moduleID)
    {
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          return getLastTemplateDepthTimestamp(devID);
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,devID,"acquisitionGetDepthTimestamp");
    return 0;
}


char * acquisitionGetColorFrame(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
  switch (moduleID)
    {
      #if USE_V4L2
      case V4L2_ACQUISITION_MODULE    :
            return getV4L2ColorPixels(devID);
      break;
      case V4L2STEREO_ACQUISITION_MODULE    :
            return getV4L2StereoColorPixels(devID);
      break;
      #endif
      case OPENGL_ACQUISITION_MODULE    :
        #if USE_OPENGL
          return getOpenGLColorPixels(devID);
        #endif
      break;
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          return getTemplateColorPixels(devID);
        #endif
      break;
      case FREENECT_ACQUISITION_MODULE:
        #if USE_FREENECT
          return getFreenectColorPixels(devID);
        #endif
      break;
      case OPENNI1_ACQUISITION_MODULE :
        #if USE_OPENNI1
            return getOpenNI1ColorPixels(devID);
        #endif
      break;
      case OPENNI2_ACQUISITION_MODULE :
        #if USE_OPENNI2
            return getOpenNI2ColorPixels(devID);
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,devID,"acquisitionGetColorFrame");
    return 0;
}

unsigned int acquisitionCopyColorFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,char * mem,unsigned int memlength)
{
  char * color = acquisitionGetColorFrame(moduleID,devID);
  if (color==0) { return 0; }
  unsigned int width , height , channels , bitsperpixel;
  acquisitionGetColorFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);
  unsigned int copySize = width*height*channels*(bitsperpixel/8);
  memcpy(mem,color,copySize);
  return copySize;
}


unsigned int acquisitionCopyColorFramePPM(ModuleIdentifier moduleID,DeviceIdentifier devID,char * mem,unsigned int memlength)
{
  char * color = acquisitionGetColorFrame(moduleID,devID);
  if (color==0) { return 0; }
  unsigned int width , height , channels , bitsperpixel;
  acquisitionGetColorFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);

  sprintf(mem, "P6%d %d\n%u\n", width, height , simplePow(2 ,bitsperpixel)-1);
  unsigned int payloadStart = strlen(mem);

  char * memPayload = mem + payloadStart ;
  memcpy(memPayload,color,width*height*channels*(bitsperpixel/8));

  payloadStart += width*height*channels*(bitsperpixel/8);
  return payloadStart;
}

short * acquisitionGetDepthFrame(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
  switch (moduleID)
    {
      #if USE_V4L2
      case V4L2_ACQUISITION_MODULE    :
            return getV4L2DepthPixels(devID);
      break;
      case V4L2STEREO_ACQUISITION_MODULE    :
            return getV4L2StereoDepthPixels(devID);
      break;
      #endif
      case OPENGL_ACQUISITION_MODULE    :
        #if USE_OPENGL
          return (short*) getOpenGLDepthPixels(devID);
        #endif
      break;
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          return (short*) getTemplateDepthPixels(devID);
        #endif
      break;
      case FREENECT_ACQUISITION_MODULE:
        #if USE_FREENECT
          return (short*) getFreenectDepthPixels(devID);
        #endif
      break;
      case OPENNI1_ACQUISITION_MODULE :
        #if USE_OPENNI1
            return (short*) getOpenNI1DepthPixels(devID);
        #endif
      break;
      case OPENNI2_ACQUISITION_MODULE :
        #if USE_OPENNI2
            return (short*) getOpenNI2DepthPixels(devID);
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,devID,"acquisitionGetDepthFrame");
    return 0;
}


unsigned int acquisitionCopyDepthFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,short * mem,unsigned int memlength)
{
  short * depth = acquisitionGetDepthFrame(moduleID,devID);
  if (depth==0) { return 0; }
  unsigned int width , height , channels , bitsperpixel;
  acquisitionGetDepthFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);
  unsigned int copySize = width*height*channels*(bitsperpixel/8);
  memcpy(mem,depth,copySize);
  return copySize;
}


unsigned int acquisitionCopyDepthFramePPM(ModuleIdentifier moduleID,DeviceIdentifier devID,short * mem,unsigned int memlength)
{
  short * depth = acquisitionGetDepthFrame(moduleID,devID);
  if (depth==0) { return 0; }

  unsigned int width , height , channels , bitsperpixel;
  acquisitionGetDepthFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);

  sprintf((char*) mem, "P5%d %d\n%u\n", width, height , simplePow(2 ,bitsperpixel)-1);
  unsigned int payloadStart = strlen((char*) mem);

  short * memPayload = mem + payloadStart ;
  memcpy(memPayload,depth,width*height*channels*(bitsperpixel/8));

  payloadStart += width*height*channels*(bitsperpixel/8);
  return payloadStart;
}


int acquisitionGetDepth3DPointAtXY(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int x2d, unsigned int y2d , float *x, float *y , float *z  )
{
    short * depthFrame = acquisitionGetDepthFrame(moduleID,devID);
    if (depthFrame == 0 ) { MeaningfullWarningMessage(moduleID,devID,"acquisitionGetDepth3DPointAtXY , getting depth frame"); return 0; }

    unsigned int width; unsigned int height; unsigned int channels; unsigned int bitsperpixel;
    if (! acquisitionGetDepthFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel) ) {  MeaningfullWarningMessage(moduleID,devID,"acquisitionGetDepth3DPointAtXY getting depth frame dims"); return 0; }

    if ( (x2d>=width) || (y2d>=height) ) { MeaningfullWarningMessage(moduleID,devID,"acquisitionGetDepth3DPointAtXY incorrect 2d x,y coords"); return 0; }

    float cx = width / 2;
    float cy = height/ 2;
    short * depthValue = depthFrame + (y2d * width + x2d );
    *z = * depthValue;
    *x = (x2d - cx) * (*z + minDistance) * scaleFactor * (width/height) ;
    *y = (y2d - cy) * (*z + minDistance) * scaleFactor;

    return 1;
}



int acquisitionGetColorFrameDimensions(ModuleIdentifier moduleID,DeviceIdentifier devID ,
                                       unsigned int * width , unsigned int * height , unsigned int * channels , unsigned int * bitsperpixel )
{
  if ( (width==0)||(height==0)||(channels==0)||(bitsperpixel==0) )
    {
        fprintf(stderr,"acquisitionGetColorFrameDimensions called with invalid arguments .. \n");
        return 0;
    }

  switch (moduleID)
    {
      #if USE_V4L2
      case V4L2_ACQUISITION_MODULE    :
          *width = getV4L2ColorWidth(devID);
          *height = getV4L2ColorHeight(devID);
          *channels = getV4L2ColorChannels(devID);
          *bitsperpixel = getV4L2ColorBitsPerPixel(devID);
          return 1;
      break;
      case V4L2STEREO_ACQUISITION_MODULE    :
          *width = getV4L2StereoColorWidth(devID);
          *height = getV4L2StereoColorHeight(devID);
          *channels = getV4L2StereoColorChannels(devID);
          *bitsperpixel = getV4L2StereoColorBitsPerPixel(devID);
          return 1;
      break;
      #endif
      case OPENGL_ACQUISITION_MODULE    :
        #if USE_OPENGL
          *width = getOpenGLColorWidth(devID);
          *height = getOpenGLColorHeight(devID);
          *channels = getOpenGLColorChannels(devID);
          *bitsperpixel = getOpenGLColorBitsPerPixel(devID);
          return 1;
        #endif
      break;
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          *width = getTemplateColorWidth(devID);
          *height = getTemplateColorHeight(devID);
          *channels = getTemplateColorChannels(devID);
          *bitsperpixel = getTemplateColorBitsPerPixel(devID);
          return 1;
        #endif
      break;
      case FREENECT_ACQUISITION_MODULE:
        #if USE_FREENECT
          *width = getFreenectColorWidth(devID);
          *height = getFreenectColorHeight(devID);
          *channels = getFreenectColorChannels(devID);
          *bitsperpixel = getFreenectColorBitsPerPixel(devID);
          return 1;
        #endif
      break;
      case OPENNI1_ACQUISITION_MODULE :
        #if USE_OPENNI1
          *width = getOpenNI1ColorWidth(devID);
          *height = getOpenNI1ColorHeight(devID);
          *channels = getOpenNI1ColorChannels(devID);
          *bitsperpixel = getOpenNI1ColorBitsPerPixel(devID);
          return 1;
        #endif
      break;
      case OPENNI2_ACQUISITION_MODULE :
        #if USE_OPENNI2
          *width = getOpenNI2ColorWidth(devID);
          *height = getOpenNI2ColorHeight(devID);
          *channels = getOpenNI2ColorChannels(devID);
          *bitsperpixel = getOpenNI2ColorBitsPerPixel(devID);
          return 1;
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,devID,"acquisitionGetColorFrameDimensions");
    return 0;
}



int acquisitionGetDepthFrameDimensions(ModuleIdentifier moduleID,DeviceIdentifier devID ,
                                       unsigned int * width , unsigned int * height , unsigned int * channels , unsigned int * bitsperpixel )
{
  if ( (width==0)||(height==0)||(channels==0)||(bitsperpixel==0) )
    {
        fprintf(stderr,"acquisitionGetDepthFrameDimensions called with invalid arguments .. \n");
        return 0;
    }

  switch (moduleID)
    {
      #if USE_V4L2
      case V4L2_ACQUISITION_MODULE    :
          *width = getV4L2DepthWidth(devID);
          *height = getV4L2DepthHeight(devID);
          *channels = getV4L2DepthChannels(devID);
          *bitsperpixel = getV4L2DepthBitsPerPixel(devID);
          return 1;
      break;
      case V4L2STEREO_ACQUISITION_MODULE    :
          *width = getV4L2StereoDepthWidth(devID);
          *height = getV4L2StereoDepthHeight(devID);
          *channels = getV4L2StereoDepthChannels(devID);
          *bitsperpixel = getV4L2StereoDepthBitsPerPixel(devID);
          return 1;
      break;
      #endif
      case OPENGL_ACQUISITION_MODULE    :
        #if USE_OPENGL
          *width = getOpenGLDepthWidth(devID);
          *height = getOpenGLDepthHeight(devID);
          *channels = getOpenGLDepthChannels(devID);
          *bitsperpixel = getOpenGLDepthBitsPerPixel(devID);
          return 1;
        #endif
      break;
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          *width = getTemplateDepthWidth(devID);
          *height = getTemplateDepthHeight(devID);
          *channels = getTemplateDepthChannels(devID);
          *bitsperpixel = getTemplateDepthBitsPerPixel(devID);
          return 1;
        #endif
      break;
      case FREENECT_ACQUISITION_MODULE:
        #if USE_FREENECT
          *width = getFreenectDepthWidth(devID);
          *height = getFreenectDepthHeight(devID);
          *channels = getFreenectDepthChannels(devID);
          *bitsperpixel = getFreenectDepthBitsPerPixel(devID);
          return 1;
        #endif
      break;
      case OPENNI1_ACQUISITION_MODULE :
        #if USE_OPENNI1
          *width = getOpenNI1DepthWidth(devID);
          *height = getOpenNI1DepthHeight(devID);
          *channels = getOpenNI1DepthChannels(devID);
          *bitsperpixel = getOpenNI1DepthBitsPerPixel(devID);
          return 1;
        #endif
      break;
      case OPENNI2_ACQUISITION_MODULE :
        #if USE_OPENNI2
          *width = getOpenNI2DepthWidth(devID);
          *height = getOpenNI2DepthHeight(devID);
          *channels = getOpenNI2DepthChannels(devID);
          *bitsperpixel = getOpenNI2DepthBitsPerPixel(devID);
          return 1;
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,devID,"acquisitionGetDepthFrameDimensions");
    return 0;
}


 int acquisitionMapDepthToRGB(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
    switch (moduleID)
    {
      case OPENGL_ACQUISITION_MODULE    :
        #if USE_OPENGL
          return 0;
        #endif
      break;
      case TEMPLATE_ACQUISITION_MODULE: /*TEMPLATE MODULE DOESNT MAP ANYTHING TO ANYTHING :P */ break;
      case FREENECT_ACQUISITION_MODULE:
        #if USE_FREENECT
          return  mapFreenectDepthToRGB(devID);
        #endif
      break;
      case OPENNI1_ACQUISITION_MODULE :
        #if USE_OPENNI1
          return  mapOpenNI1DepthToRGB(devID);
        #endif
      break;
      case OPENNI2_ACQUISITION_MODULE :
        #if USE_OPENNI2
          return mapOpenNI2DepthToRGB(devID);
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,devID,"acquisitionMapDepthToRGB");
    return 0;
}


 int acquisitionMapRGBToDepth(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
    switch (moduleID)
    {
      case V4L2_ACQUISITION_MODULE    :   break;
      case OPENGL_ACQUISITION_MODULE    :  break;
      case TEMPLATE_ACQUISITION_MODULE: /*TEMPLATE MODULE DOESNT MAP ANYTHING TO ANYTHING :P */ break;
      case FREENECT_ACQUISITION_MODULE:   break;
      case OPENNI1_ACQUISITION_MODULE :
        #if USE_OPENNI1
          return  mapOpenNI1RGBToDepth(devID);
        #endif
      break;
      case OPENNI2_ACQUISITION_MODULE :
        #if USE_OPENNI2
          return mapOpenNI2RGBToDepth(devID);
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,devID,"acquisitionMapRGBToDepth");
    return 0;
}



double acqusitionGetColorFocalLength(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
    switch (moduleID)
    {
      case V4L2_ACQUISITION_MODULE    :   break;
      case OPENGL_ACQUISITION_MODULE    :
        #if USE_OPENGL
          return getOpenGLColorFocalLength(devID);
        #endif
      break;
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          return  getTemplateColorFocalLength(devID);
        #endif
      break;
      case FREENECT_ACQUISITION_MODULE:   break;
      case OPENNI1_ACQUISITION_MODULE :
        #if USE_OPENNI1
          return  getOpenNI1ColorFocalLength(devID);
        #endif
      break;
      case OPENNI2_ACQUISITION_MODULE :
        #if USE_OPENNI2
          return  getOpenNI2ColorFocalLength(devID);
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,devID,"acqusitionGetColorFocalLength");
    return 0.0;
}

double acqusitionGetColorPixelSize(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
    switch (moduleID)
    {
      case V4L2_ACQUISITION_MODULE    :   break;
      case OPENGL_ACQUISITION_MODULE    :  break;
        #if USE_OPENGL
          return getOpenGLColorPixelSize(devID);
        #endif
      break;
      case TEMPLATE_ACQUISITION_MODULE: /*TEMPLATE MODULE DOESNT MAP ANYTHING TO ANYTHING :P */
        #if USE_TEMPLATE
          return  getTemplateColorPixelSize(devID);
        #endif
      break;
      case FREENECT_ACQUISITION_MODULE:   break;
      case OPENNI1_ACQUISITION_MODULE :
        #if USE_OPENNI1
          return  getOpenNI1ColorPixelSize(devID);
        #endif
      break;
      case OPENNI2_ACQUISITION_MODULE :
        #if USE_OPENNI2
          return  getOpenNI2ColorPixelSize(devID);
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,devID,"acqusitionGetColorPixelSize");
    return 0.0;
}



double acqusitionGetDepthFocalLength(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
    switch (moduleID)
    {
      case V4L2_ACQUISITION_MODULE    :   break;
      case OPENGL_ACQUISITION_MODULE    :
        #if USE_OPENGL
          return getOpenGLDepthFocalLength(devID);
        #endif
      break;
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          return  getTemplateDepthFocalLength(devID);
        #endif
      break;
      case FREENECT_ACQUISITION_MODULE:   break;
      case OPENNI1_ACQUISITION_MODULE :
        #if USE_OPENNI1
          return  getOpenNI1DepthFocalLength(devID);
        #endif
      break;
      case OPENNI2_ACQUISITION_MODULE :
        #if USE_OPENNI2
          return  getOpenNI2DepthFocalLength(devID);
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,devID,"acqusitionGetFocalLength");
    return 0.0;
}

double acqusitionGetDepthPixelSize(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
    switch (moduleID)
    {
      case V4L2_ACQUISITION_MODULE    :   break;
      case OPENGL_ACQUISITION_MODULE    :  break;
        #if USE_OPENGL
          return getOpenGLDepthPixelSize(devID);
        #endif
      break;
      case TEMPLATE_ACQUISITION_MODULE: /*TEMPLATE MODULE DOESNT MAP ANYTHING TO ANYTHING :P */
        #if USE_TEMPLATE
          return  getTemplateDepthPixelSize(devID);
        #endif
      break;
      case FREENECT_ACQUISITION_MODULE:   break;
      case OPENNI1_ACQUISITION_MODULE :
        #if USE_OPENNI1
          return  getOpenNI1DepthPixelSize(devID);
        #endif
      break;
      case OPENNI2_ACQUISITION_MODULE :
        #if USE_OPENNI2
          return  getOpenNI2DepthPixelSize(devID);
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,devID,"acqusitionGetPixelSize");
    return 0.0;
}

