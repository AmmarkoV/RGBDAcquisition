#include "Acquisition.h"
#include "acquisition_setup.h"

#include <stdio.h>
#include <stdlib.h>

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

int saveRawImageToFile(char * filename,char * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel)
{
    if(pixels==0) { fprintf(stderr,"saveRawImageToFile(%s) called for an unallocated (empty) frame , will not write any file output\n",filename); return 0; }
    FILE *fd=0;
    fd = fopen(filename,"wb");

    if (bitsperpixel>16) fprintf(stderr,"PNM does not support more than 2 bytes per pixel..!\n");
    if (fd!=0)
    {
        unsigned int n;
        if (channels==3) fprintf(fd, "P6\n");
        else if (channels==1) fprintf(fd, "P5\n");
        else
        {
            fprintf(stderr,"Invalid channels arg (%u) for SaveRawImageToFile\n",channels);
            return 1;
        }

        fprintf(fd, "%d %d\n%u\n", width, height , simplePow(2 ,bitsperpixel)-1);

        float tmp_n = (float) bitsperpixel/ 8;
        tmp_n = tmp_n *  width * height * channels ;
        n = (unsigned int) tmp_n;

        fwrite(pixels, 1 , n , fd);
        //fwrite(pixels, 1 , n , fd);
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


int acquisitionGetModulesCount()
{
  unsigned int modules_linked = 0;

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
          if (strcasecmp("OPENNI1",moduleName)==0 )  { moduleID = OPENNI1_ACQUISITION_MODULE;  } else
          if (strcasecmp("OPENNI2",moduleName)==0 )  { moduleID = OPENNI2_ACQUISITION_MODULE;  } else
          if (strcasecmp("OPENGL",moduleName)==0 )   { moduleID = OPENGL_ACQUISITION_MODULE;   } else
          if (strcasecmp("TEMPLATE",moduleName)==0 )  { moduleID = TEMPLATE_ACQUISITION_MODULE; }
   return moduleID;
}


char * getModuleStringName(ModuleIdentifier moduleID)
{
  switch (moduleID)
    {
      case V4L2_ACQUISITION_MODULE    :  return (char*) "V4L2 MODULE"; break;
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
      case V4L2_ACQUISITION_MODULE    :   break;
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
      case V4L2_ACQUISITION_MODULE    :   break;
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
      case V4L2_ACQUISITION_MODULE    :   break;
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
      case V4L2_ACQUISITION_MODULE    :   break;
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
int acquisitionOpenDevice(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int width,unsigned int height,unsigned int framerate)
{
    switch (moduleID)
    {
      case V4L2_ACQUISITION_MODULE    :   break;
      case OPENGL_ACQUISITION_MODULE    :
        #if USE_OPENGL
          return createOpenGLDevice(devID,width,height,framerate);
        #endif
      break;
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          return createTemplateDevice(devID,width,height,framerate);
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
      case V4L2_ACQUISITION_MODULE    :   break;
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


 int acquisitionSnapFrames(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
    switch (moduleID)
    {
      case V4L2_ACQUISITION_MODULE    :   break;
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
    switch (moduleID)
    {
      case V4L2_ACQUISITION_MODULE    :   break;
      case OPENGL_ACQUISITION_MODULE    :
        #if USE_OPENGL
          return saveRawImageToFile(filename,getOpenGLColorPixels(devID),getOpenGLColorWidth(devID),getOpenGLColorHeight(devID),getOpenGLColorChannels(devID),getOpenGLColorBitsPerPixel(devID));
        #endif
      break;
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          return saveRawImageToFile(filename,getTemplateColorPixels(devID),getTemplateColorWidth(devID),getTemplateColorHeight(devID) ,getTemplateColorChannels(devID),getTemplateColorBitsPerPixel(devID));
        #endif
      break;
      case FREENECT_ACQUISITION_MODULE:
        #if USE_FREENECT
          return saveRawImageToFile(filename,getFreenectColorPixels(devID),getFreenectColorWidth(devID),getFreenectColorHeight(devID) ,getFreenectColorChannels(devID),getFreenectColorBitsPerPixel(devID));
        #endif
      break;
      case OPENNI1_ACQUISITION_MODULE :
        #if USE_OPENNI1
          return saveRawImageToFile(filename,getOpenNI1ColorPixels(devID),getOpenNI1ColorWidth(devID),getOpenNI1ColorHeight(devID) ,getOpenNI1ColorChannels(devID),getOpenNI1ColorBitsPerPixel(devID));
        #endif
      break;
      case OPENNI2_ACQUISITION_MODULE :
        #if USE_OPENNI2
          return saveRawImageToFile(filename,getOpenNI2ColorPixels(devID),getOpenNI2ColorWidth(devID),getOpenNI2ColorHeight(devID) ,getOpenNI2ColorChannels(devID),getOpenNI2ColorBitsPerPixel(devID));
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,devID,"acquisitionSaveColorFrame");
    return 0;
}

 int acquisitionSaveDepthFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename)
{
    switch (moduleID)
    {
      case V4L2_ACQUISITION_MODULE    :   break;
      case OPENGL_ACQUISITION_MODULE    :
        #if USE_OPENGL
          return saveRawImageToFile(filename,(char*) getOpenGLDepthPixels(devID),getOpenGLDepthWidth(devID),getOpenGLDepthHeight(devID),getOpenGLDepthChannels(devID),getOpenGLDepthBitsPerPixel(devID));
        #endif
      break;
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          return saveRawImageToFile(filename,getTemplateDepthPixels(devID),getTemplateDepthWidth(devID),getTemplateDepthHeight(devID) ,getTemplateDepthChannels(devID),getTemplateDepthBitsPerPixel(devID));
        #endif
      break;
      case FREENECT_ACQUISITION_MODULE:
        #if USE_FREENECT
          return saveRawImageToFile(filename,getFreenectDepthPixels(devID),getFreenectDepthWidth(devID),getFreenectDepthHeight(devID) ,getFreenectDepthChannels(devID),getFreenectDepthBitsPerPixel(devID));
        #endif
      break;
      case OPENNI1_ACQUISITION_MODULE :
        #if USE_OPENNI1
            return saveRawImageToFile(filename,getOpenNI1DepthPixels(devID),getOpenNI1DepthWidth(devID),getOpenNI1DepthHeight(devID) ,getOpenNI1DepthChannels(devID),getOpenNI1DepthBitsPerPixel(devID));
        #endif
      break;
      case OPENNI2_ACQUISITION_MODULE :
        #if USE_OPENNI2
            return saveRawImageToFile(filename,getOpenNI2DepthPixels(devID),getOpenNI2DepthWidth(devID),getOpenNI2DepthHeight(devID) ,getOpenNI2DepthChannels(devID),getOpenNI2DepthBitsPerPixel(devID));
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,devID,"acquisitionSaveDepthFrame");
    return 0;
}



char * acquisitionGetColorFrame(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
  switch (moduleID)
    {
      case V4L2_ACQUISITION_MODULE    :   break;
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


short * acquisitionGetDepthFrame(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
  switch (moduleID)
    {
      case V4L2_ACQUISITION_MODULE    :   break;
      case OPENGL_ACQUISITION_MODULE    :
        #if USE_OPENGL
          return getOpenGLDepthPixels(devID);
        #endif
      break;
      case TEMPLATE_ACQUISITION_MODULE:
        #if USE_TEMPLATE
          return getTemplateDepthPixels(devID);
        #endif
      break;
      case FREENECT_ACQUISITION_MODULE:
        #if USE_FREENECT
          return getFreenectDepthPixels(devID);
        #endif
      break;
      case OPENNI1_ACQUISITION_MODULE :
        #if USE_OPENNI1
            return getOpenNI1DepthPixels(devID);
        #endif
      break;
      case OPENNI2_ACQUISITION_MODULE :
        #if USE_OPENNI2
            return getOpenNI2DepthPixels(devID);
        #endif
      break;
    };
    MeaningfullWarningMessage(moduleID,devID,"acquisitionGetDepthFrame");
    return 0;
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
      case V4L2_ACQUISITION_MODULE    :   break;
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
      case V4L2_ACQUISITION_MODULE    :   break;
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
      case V4L2_ACQUISITION_MODULE    :   break;
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

