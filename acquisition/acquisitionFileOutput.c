
#include "Acquisition.h"
#include "acquisition_setup.h"
#include "pluginLinker.h"
#include "processorLinker.h"
#include "acquisitionFileOutput.h"




#include "../tools/Timers/timer.h"
#include "../tools/OperatingSystem/OperatingSystem.h"
#include "../tools/Primitives/modules.h"



   #if ENABLE_LOCATION_SERVICE
    #include "../tools/LocationServices/locationService.h"
   #endif // ENABLE_LOCATION_SERVICE



#if (ENABLE_PNG && ENABLE_JPG)
  #include "../tools/Codecs/codecs.h"
  #define USE_CODEC_LIBRARY 1 //Not Using the codec library really simplifies build things but we lose png/jpg formats
  #warning "Acquisition Library Compiling With JPG/PNG output"
#endif // ENABLE_PNG

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
//This should probably pass on to each of the sub modules
float minDistance = -10;
float scaleFactor = 0.0021;



unsigned int _acfo_simplePow(unsigned int base,unsigned int exp)
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


int _acfo_savePCD_PointCloud(char * filename ,unsigned short * depthFrame ,unsigned char * colorFrame , unsigned int width , unsigned int height , float cx , float cy , float fx , float fy )
{
    if(depthFrame==0) { fprintf(stderr,"saveToPCD_PointCloud(%s) called for an unallocated (empty) depth frame , will not write any file output\n",filename); return 0; }
    if(colorFrame==0) { fprintf(stderr,"saveToPCD_PointCloud(%s) called for an unallocated (empty) color frame , will not write any file output\n",filename); return 0; }

    FILE *fd=0;
    fd = fopen(filename,"wb");
    if (fd!=0)
    {
        fprintf(fd, "# .PCD v.7 - Point Cloud Data file format\n");
        fprintf(fd, "FIELDS x y z rgb\n");
        fprintf(fd, "SIZE 4 4 4 4\n");
        fprintf(fd, "TYPE F F F U\n");
        fprintf(fd, "COUNT 1\n");
        fprintf(fd, "WIDTH %u\n",width);
        fprintf(fd, "HEIGHT %u\n",height);
        fprintf(fd, "POINTS %u\n",width*height);
        fprintf(fd, "DATA ascii\n");

        unsigned short * depthPTR = depthFrame;
        unsigned char  * colorPTR = colorFrame;

        unsigned int px=0,py=0;
        float x=0.0,y=0.0,z=0.0;
        unsigned char * r , * b , * g;
        unsigned int rgb=0;

        //Restart Depth
        depthPTR = depthFrame;


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



int _acfo_savePCD_PointCloudNoEmpty(char * filename ,unsigned short * depthFrame ,unsigned char * colorFrame , unsigned int width , unsigned int height , float cx , float cy , float fx , float fy )
{
    if(depthFrame==0) { fprintf(stderr,"saveToPCD_PointCloud(%s) called for an unallocated (empty) depth frame , will not write any file output\n",filename); return 0; }
    if(colorFrame==0) { fprintf(stderr,"saveToPCD_PointCloud(%s) called for an unallocated (empty) color frame , will not write any file output\n",filename); return 0; }

    FILE *fd=0;
    fd = fopen(filename,"wb");
    if (fd!=0)
    {
        fprintf(fd, "# .PCD v.7 - Point Cloud Data file format\n");
        fprintf(fd, "FIELDS x y z rgb\n");
        fprintf(fd, "SIZE 4 4 4 4\n");
        fprintf(fd, "TYPE F F F U\n");
        fprintf(fd, "COUNT 1\n");

        unsigned short * depthPTR = depthFrame;
        unsigned char  * colorPTR = colorFrame;

        unsigned int px=0,py=0;
        float x=0.0,y=0.0,z=0.0;
        unsigned char * r , * b , * g;
        unsigned int rgb=0;

        unsigned int totalPoints = 0;
        for (py=0; py<height; py++)
        {
         for (px=0; px<width; px++)
         {
           if (*depthPTR!=0) { ++totalPoints; }  ++depthPTR;
         }
        }
        fprintf(fd, "WIDTH %u\n",totalPoints);
        fprintf(fd, "HEIGHT %u\n",1);
        fprintf(fd, "POINTS %u\n",totalPoints);
        fprintf(fd, "DATA ascii\n");


        //Restart Depth
        depthPTR = depthFrame;


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

           if (z!=0)
              { fprintf(fd, "%0.4f %0.4f %0.4f %u\n",x,y,z,rgb); }
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


 int _acfo_acquisitionSavePCDPointCoud(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename)
{
  unsigned int width;
  unsigned int height;
  unsigned int channels;
  unsigned int bitsperpixel;
  acquisitionGetDepthFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);

  return savePCD_PointCloud(filename,acquisitionGetDepthFrame(moduleID,devID),acquisitionGetColorFrame(moduleID,devID),
                            width,height,width/2,height/2, 1.0 /*DUMMY fx*/, 1.0 /*DUMMY fy*/ );
}




int _acfo_swapEndiannessPNM(void * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel)
{
  unsigned char * traverser=(unsigned char * ) pixels;
  unsigned char * traverserSwap1=(unsigned char * ) pixels;
  unsigned char * traverserSwap2=(unsigned char * ) pixels;

  unsigned int bytesperpixel = (bitsperpixel/8);
  unsigned char * endOfMem = traverser + width * height * channels * bytesperpixel;

  unsigned char tmp ;
  while ( ( traverser < endOfMem)  )
  {
    traverserSwap1 = traverser;
    traverserSwap2 = traverser+1;

    tmp = *traverserSwap1;
    *traverserSwap1 = *traverserSwap2;
    *traverserSwap2 = tmp;

    traverser += bytesperpixel;
  }

 return 1;
}


int _acfo_acquisitionSaveRawImageToFile(char * filename,unsigned char * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel)
{
   // fprintf(stderr,"acquisitionSaveRawImageToFile(%s) called\n",filename);

    #if USE_REGULAR_BYTEORDER_FOR_PNM
     //Want Conformance to the NETPBM spec http://en.wikipedia.org/wiki/Netpbm_format#16-bit_extensions
     if (bitsperpixel==16) { swapEndiannessPNM(pixels , width , height , channels , bitsperpixel); }
    #else
      #warning "We are using Our Local Byte Order for saving files , this makes things fast but is incompatible with other PNM loaders"
    #endif // USE_REGULAR_BYTEORDER_FOR_PNM

    if ( (width==0) || (height==0) || (channels==0) || (bitsperpixel==0) ) { fprintf(stderr,"acquisitionSaveRawImageToFile(%s) called with zero dimensions\n",filename); return 0;}
    if(pixels==0) { fprintf(stderr,"acquisitionSaveRawImageToFile(%s) called for an unallocated (empty) frame , will not write any file output\n",filename); return 0; }
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

        /*
        char timeStampStr[256]={0};
        GetDateString(timeStampStr,"TIMESTAMP",1,0,0,0,0,0,0,0);
        fprintf(fd, "#%s\n", timeStampStr );*/

        fprintf(fd, "#TIMESTAMP %lu\n",GetTickCount());


        fprintf(fd, "%d %d\n%u\n", width, height , _acfo_simplePow(2 ,bitsperpixel)-1);

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



int _acfo_acquisitionSaveLocationStamp(char * filename)
{
   #if ENABLE_LOCATION_SERVICE
   if (useLocationServices)
   {
   if (locationServicesOK())
   {
    FILE *fd=0;
    fd = fopen(filename,"w");

    if (fd!=0)
    {
     fprintf(fd,"Sats(%u)\n",sattelitesUsed());
     fprintf(fd,"Alt(%0.5f)\n",getAlt());
     fprintf(fd,"Lat(%0.5f)\n",getLat());
     fprintf(fd,"Lon(%0.5f)\n",getLon());
     fprintf(fd,"Speed(%0.5f)\n",getSpeed());
     fprintf(fd,"Climb(%0.5f)\n",getClimb());
     fprintf(fd,"Bearing(%0.5f)\n",getBearing());

     fflush(fd);
     fclose(fd);
    }
    return 1;
   } else
   {
    return 0;
   }

   }
   #else
    return 0;
   #endif

return 0;
}

//Ok this is basically casting the 2 bytes of depth into 3 RGB bytes leaving one color channel off (the blue one)
//depth is casted to char to simplify things , but that adds sizeof(short) to the pointer arethemetic!
unsigned char * _acfo_convertShortDepthToRGBDepth(unsigned short * depth,unsigned int width , unsigned int height)
{
  if (depth==0)  { fprintf(stderr,"Depth is not allocated , cannot perform DepthToRGB transformation \n"); return 0; }
  unsigned char * depthPTR= (unsigned char*) depth; // This will be the traversing pointer for input
  unsigned char * depthLimit = (unsigned char*) depth + width*height * sizeof(unsigned short); //<- we use sizeof(short) because we have casted to char !


  unsigned char * outFrame = (unsigned char*) malloc(width*height*3*sizeof(unsigned char));
  if (outFrame==0) { fprintf(stderr,"Could not perform DepthToRGB transformation\nNo memory for new frame\n"); return 0; }

  unsigned char * outFramePTR = outFrame; // This will be the traversing pointer for output
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
unsigned char * _acfo_convertShortDepthToCharDepth(unsigned short * depth,unsigned int width , unsigned int height , unsigned int min_depth , unsigned int max_depth)
{
  if (depth==0)  { fprintf(stderr,"Depth is not allocated , cannot perform DepthToRGB transformation \n"); return 0; }
  unsigned short * depthPTR= depth; // This will be the traversing pointer for input
  unsigned short * depthLimit =  depth + width*height; //<- we use sizeof(short) because we have casted to char !


  unsigned char * outFrame = (unsigned char*) malloc(width*height*1*sizeof(char));
  if (outFrame==0) { fprintf(stderr,"Could not perform DepthToRGB transformation\nNo memory for new frame\n"); return 0; }

  float depth_range = max_depth-min_depth;
  if (depth_range ==0 ) { depth_range = 1; }
  float multiplier = 255 / depth_range;


  unsigned char * outFramePTR = outFrame; // This will be the traversing pointer for output
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


//Ok this is basically casting the 2 bytes of depth into 3 RGB bytes ( grayscale )
unsigned char * _acfo_convertShortDepthTo3CharDepth(unsigned short * depth,unsigned int width , unsigned int height , unsigned int min_depth , unsigned int max_depth)
{
  if (depth==0)  { fprintf(stderr,"Depth is not allocated , convertShortDepthTo3CharDepth cannot continue \n"); return 0; }
  unsigned short * depthPTR= depth; // This will be the traversing pointer for input
  unsigned short * depthLimit =  depth + width*height; //<- we use sizeof(short) because we have casted to char !


  unsigned char * outFrame = (unsigned char*) malloc(width*height*3*sizeof(unsigned char));
  if (outFrame==0) { fprintf(stderr,"Could not perform DepthToRGB transformation\nNo memory for new frame\n"); return 0; }

  float depth_range = max_depth-min_depth;
  if (depth_range ==0 ) { depth_range = 1; }
  float multiplier = 255 / depth_range;


  unsigned char * outFramePTR = outFrame; // This will be the traversing pointer for output
  while ( depthPTR<depthLimit )
  {
     unsigned int scaled = (unsigned int) (*depthPTR) * multiplier;
     unsigned char scaledChar = (unsigned char) scaled;
     * outFramePTR = scaledChar; ++outFramePTR;
     * outFramePTR = scaledChar; ++outFramePTR;
     * outFramePTR = scaledChar; ++outFramePTR;

     ++depthPTR;
  }
 return outFrame;
}









int _acfo_acquisitionSaveColorFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename, int compress)
{
    printCall(moduleID,devID,"acquisitionSaveColorFrame", __FILE__, __LINE__);
    char filenameFull[2048]={0};
    int retres =0;

    #if USE_REGULAR_BYTEORDER_FOR_PNM
     sprintf(filenameFull,"%s.cpnm",filename);
    #else
     sprintf(filenameFull,"%s.pnm",filename);
    #endif


         if (
              (*plugins[moduleID].getColorPixels!=0) && (*plugins[moduleID].getColorWidth!=0) && (*plugins[moduleID].getColorHeight!=0) &&
              (*plugins[moduleID].getColorChannels!=0) && (*plugins[moduleID].getColorBitsPerPixel!=0)
            )
         {
             #if USE_CODEC_LIBRARY
             //fprintf(stderr,"using color codec library ( compress %u ) \n",compress);
             if (compress)
             {
               struct Image * img = createImageUsingExistingBuffer( (*plugins[moduleID].getColorWidth)       (devID),
                                                                    (*plugins[moduleID].getColorHeight)      (devID),
                                                                    (*plugins[moduleID].getColorChannels)    (devID),
                                                                    (*plugins[moduleID].getColorBitsPerPixel)(devID),
                                                                    acquisitionGetColorFrame(moduleID,devID)
                                                                  );

               sprintf(filenameFull,"%s.jpg",filename);
               writeImageFile(img,JPG_CODEC,filenameFull);

               img->pixels=0;
               destroyImage(img);
             } else
             #endif // USE_CODEC_LIBRARY
             {
                retres = acquisitionSaveRawImageToFile(
                                            filenameFull,
                                            acquisitionGetColorFrame(moduleID,devID)        ,
                                            (*plugins[moduleID].getColorWidth)       (devID),
                                            (*plugins[moduleID].getColorHeight)      (devID),
                                            (*plugins[moduleID].getColorChannels)    (devID),
                                            (*plugins[moduleID].getColorBitsPerPixel)(devID)
                                           );

           // V4L2 Specific JPS compression ------------------------------------------------------------------------------------------------------------------------------
           if ( (retres) && (moduleID==V4L2STEREO_ACQUISITION_MODULE) )
              {  //V4L2Stereo images are huge so until we fix jpeg compression for all templates ( already there but there are some managment decisions to be made :P )
                 //we do a simple hack here :p
                 char convertToJPEGString[4096]={0};
                 snprintf(convertToJPEGString , 4096, "convert %s.pnm %s.jpg && rm  %s.pnm && mv %s.jpg %s.jps",filename,filename,filename,filename,filename);
                 int i = system(convertToJPEGString);
                 if (i==0) { fprintf(stderr,"Success converting to jpeg\n"); } else
                           { fprintf(stderr,"Failure converting to jpeg\n"); }
              }
           // V4L2 Specific JPS compression ------------------------------------------------------------------------------------------------------------------------------

             }





              #if ENABLE_LOCATION_SERVICE
              if (useLocationServices)
              {
               char timestampFilename[4096]={0};
               snprintf(timestampFilename, 4096, "%s_loc.txt",filename);
               if (!acquisitionSaveLocationStamp(timestampFilename))
               {
                 fprintf(stderr,"Could not save location stamp [ %s ] \n",filename);
               }
              }
              #endif // ENABLE_LOCATION_SERVICE

            return retres;
         }


    MeaningfullWarningMessage(moduleID,devID,"acquisitionSaveColorFrame");
    return 0;
}




int _acfo_acquisitionSaveDepthFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename, int compress)
{
    printCall(moduleID,devID,"acquisitionSaveDepthFrame", __FILE__, __LINE__);
    char filenameFull[2048]={0};
    int retres=0;

    #if USE_REGULAR_BYTEORDER_FOR_PNM
     sprintf(filenameFull,"%s.cpnm",filename);
    #else
     sprintf(filenameFull,"%s.pnm",filename);
    #endif

          if (
              (*plugins[moduleID].getDepthPixels!=0) && (*plugins[moduleID].getDepthWidth!=0) && (*plugins[moduleID].getDepthHeight!=0) &&
              (*plugins[moduleID].getDepthChannels!=0) && (*plugins[moduleID].getDepthBitsPerPixel!=0)
             )
         {

             #if USE_CODEC_LIBRARY
             //fprintf(stderr,"using depth codec library ( compress %u ) \n",compress);
             if (compress)
             {
               struct Image * img = createImageUsingExistingBuffer( (*plugins[moduleID].getDepthWidth)       (devID),
                                                                    (*plugins[moduleID].getDepthHeight)      (devID),
                                                                    (*plugins[moduleID].getDepthChannels)    (devID),
                                                                    (*plugins[moduleID].getDepthBitsPerPixel)(devID),
                                                                    acquisitionGetDepthFrame(moduleID,devID)
                                                                  );

               //Want Conformance to the NETPBM spec http://en.wikipedia.org/wiki/Netpbm_format#16-bit_extensions
              if ((*plugins[moduleID].getDepthBitsPerPixel)(devID)==16)
                 { swapEndiannessPNM(
                                     acquisitionGetDepthFrame(moduleID,devID) ,
                                     (*plugins[moduleID].getDepthWidth)       (devID),
                                     (*plugins[moduleID].getDepthHeight)      (devID),
                                     (*plugins[moduleID].getDepthChannels)    (devID),
                                     (*plugins[moduleID].getDepthBitsPerPixel)(devID)
                                    );
                 }

               sprintf(filenameFull,"%s.png",filename);
               writeImageFile(img,PNG_CODEC,filenameFull);

               img->pixels=0;
               destroyImage(img);
             } else
             #endif //USE_CODEC_LIBRARY
             {
              retres=acquisitionSaveRawImageToFile(
                                      filenameFull,
                                      // (*plugins[moduleID].getDepthPixels)      (devID),
                                      (unsigned char*) acquisitionGetDepthFrame(moduleID,devID)  ,
                                      (*plugins[moduleID].getDepthWidth)       (devID),
                                      (*plugins[moduleID].getDepthHeight)      (devID),
                                      (*plugins[moduleID].getDepthChannels)    (devID),
                                      (*plugins[moduleID].getDepthBitsPerPixel)(devID)
                                     );
              return retres;
             }
         }

    MeaningfullWarningMessage(moduleID,devID,"acquisitionSaveDepthFrame");
    return 0;
}



int _acfo_acquisitionSaveColoredDepthFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename)
{
    printCall(moduleID,devID,"acquisitionSaveColoredDepthFrame", __FILE__, __LINE__);

    char filenameFull[1024]={0};


    #if USE_REGULAR_BYTEORDER_FOR_PNM
     sprintf(filenameFull,"%s.cpnm",filename);
    #else
     sprintf(filenameFull,"%s.pnm",filename);
    #endif

    unsigned int width = 0 ;
    unsigned int height = 0 ;
    unsigned int channels = 0 ;
    unsigned int bitsperpixel = 0 ;
    unsigned short * inFrame = 0;
    unsigned char * outFrame = 0 ;

    inFrame = acquisitionGetDepthFrame(moduleID,devID);
    if (inFrame!=0)
      {
       acquisitionGetDepthFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);
       outFrame = convertShortDepthToRGBDepth(inFrame,width,height);
       if (outFrame!=0)
        {
         acquisitionSaveRawImageToFile(filenameFull,outFrame,width,height,3,8);
         free(outFrame);
         return 1;
        }
      }

    MeaningfullWarningMessage(moduleID,devID,"acquisitionSaveColoredDepthFrame");
    return 0;
}


int _acfo_acquisitionSaveDepthFrame1C(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename)
{
    printCall(moduleID,devID,"acquisitionSaveColoredDepthFrame", __FILE__, __LINE__);

    char filenameFull[1024]={0};

    #if USE_REGULAR_BYTEORDER_FOR_PNM
     sprintf(filenameFull,"%s.cpnm",filename);
    #else
     sprintf(filenameFull,"%s.pnm",filename);
    #endif

    unsigned int width = 0 ;
    unsigned int height = 0 ;
    unsigned int channels = 0 ;
    unsigned int bitsperpixel = 0 ;
    unsigned short * inFrame = 0;
    unsigned char * outFrame = 0 ;

    inFrame = acquisitionGetDepthFrame(moduleID,devID);
    if (inFrame!=0)
      {
       acquisitionGetDepthFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);
       outFrame = convertShortDepthToCharDepth(inFrame,width,height,0,7000);
       if (outFrame!=0)
        {
         acquisitionSaveRawImageToFile(filenameFull,outFrame,width,height,1,8);
         free(outFrame);
         return 1;
        }
      }

    MeaningfullWarningMessage(moduleID,devID,"acquisitionSaveColoredDepthFrame");
    return 0;
}

