#include "templateAcquisitionHelper.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define PRINT_COMMENTS 1

#if ENABLE_PNG
  #define USE_CODEC_LIBRARY 1 //Not Using the codec library really simplifies build things but we lose png/jpg formats
#endif // ENABLE_PNG

#if ENABLE_JPG
  #define USE_CODEC_LIBRARY 1 //Not Using the codec library really simplifies build things but we lose png/jpg formats
#endif // ENABLE_PNG

 #if USE_CODEC_LIBRARY
  #include "../tools/Codecs/codecs.h"
 #else
  //#error "No codec library, will not compile.."
 #endif // USE_CODEC_LIBRARY


#define PPMREADBUFLEN 256

int _tacq_directoryExists(const char* folder)
{
 if (folder==0) { return 0; }
 struct stat sb;

 //fprintf(stderr,"Checking if directory (%s) exists : ",folder);
 if (stat(folder, &sb) == 0 && S_ISDIR(sb.st_mode))
    {
        //fprintf(stderr,"yes\n");
        return 1;
    }
 //fprintf(stderr,"no\n");
 return 0;
}


int fileExists(char * filename)
{
 if (filename==0) { return 0; }
 //fprintf(stderr,"Checking if file (%s) exists : ",filename);
 FILE *fp = fopen(filename,"r");
 if( fp ) { /* exists */
            fclose(fp);
            //fprintf(stderr,"yes\n");
            return 1;
          }
 /* doesnt exist */
 //fprintf(stderr,"no\n");
 return 0;
}

#ifndef USE_CODEC_LIBRARY

int swapEndiannessInternalPNM(void * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel)
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



unsigned char * ReadPNMInternal(unsigned char * buffer , char * filename,unsigned int *width,unsigned int *height,unsigned long * timestamp)
{
    //See http://en.wikipedia.org/wiki/Portable_anymap#File_format_description for this simple and useful format
    unsigned char * pixels=buffer;
    FILE *pf=0;
    pf = fopen(filename,"rb");

    if (pf!=0 )
    {
        *width=0; *height=0; *timestamp=0;
        unsigned int bytesPerPixel=0;
        unsigned int channels=0;
        char buf[PPMREADBUFLEN]={0};
        char *t;
        unsigned int w=0, h=0, d=0;
        int r=0 , z=0;

        t = fgets(buf, PPMREADBUFLEN, pf);
        if (t == 0) { return buffer; }

        if ( strncmp(buf,"P6\n", 3) == 0 ) { channels=3; } else
        if ( strncmp(buf,"P5\n", 3) == 0 ) { channels=1; } else
                                           { fprintf(stderr,"ReadPNMInternal : Could not understand/Not supported file format\n"); fclose(pf); return buffer; }
        do
        { /* Px formats can have # comments after first line */
           #if PRINT_COMMENTS
             memset(buf,0,PPMREADBUFLEN);
           #endif
           t = fgets(buf, PPMREADBUFLEN, pf);
           if (strstr(buf,"TIMESTAMP")!=0)
              {
                char * timestampPayloadStr = buf + 10;
                *timestamp = atoi(timestampPayloadStr);
              }

           if ( t == 0 ) { fclose(pf); return buffer; }
        } while ( strncmp(buf, "#", 1) == 0 );
        z = sscanf(buf, "%u %u", &w, &h);
        if ( z < 2 ) { fclose(pf); fprintf(stderr,"Incoherent dimensions received %ux%u \n",w,h); return buffer; }
        // The program fails if the first byte of the image is equal to 32. because
        // the fscanf eats the space and the image is read with some bit less
        r = fscanf(pf, "%u\n", &d);
        if (r < 1) { fprintf(stderr,"Could not understand how many bytesPerPixel there are on this image\n"); fclose(pf); return buffer; }
        if (d==255) { bytesPerPixel=1; }  else
        if (d==65535) { bytesPerPixel=2; } else
                       { fprintf(stderr,"Incoherent payload received %u bits per pixel \n",d); fclose(pf); return buffer; }


        //This is a super ninja hackish patch that fixes the case where fscanf eats one character more on the stream
        //It could be done better  ( no need to fseek ) but this will have to do for now
        //Scan for border case
           unsigned long startOfBinaryPart = ftell(pf);
           if ( fseek (pf , 0 , SEEK_END)!=0 ) { fprintf(stderr,"Could not find file size to cache client..!\nUnable to serve client\n"); fclose(pf); return 0; }
           unsigned long totalFileSize = ftell (pf); //lSize now holds the size of the file..

           //fprintf(stderr,"totalFileSize-startOfBinaryPart = %u \n",totalFileSize-startOfBinaryPart);
           //fprintf(stderr,"bytesPerPixel*channels*w*h = %u \n",bytesPerPixel*channels*w*h);
           if (totalFileSize-startOfBinaryPart < bytesPerPixel*channels*w*h )
           {
              unsigned int differenceInBytes =  bytesPerPixel*channels*w*h;
              differenceInBytes = differenceInBytes -totalFileSize + startOfBinaryPart;
              fprintf(stderr," Detected Border Case (%u) \n\n\n",differenceInBytes);
              startOfBinaryPart-=differenceInBytes;
           }
           if ( fseek (pf , startOfBinaryPart , SEEK_SET)!=0 ) { fprintf(stderr,"Could not find file size to cache client..!\nUnable to serve client\n"); fclose(pf); return 0; }
         //-----------------------
         //----------------------

        *width=w; *height=h;
        if (pixels==0) {  pixels= (unsigned char*) malloc(w*h*bytesPerPixel*channels*sizeof(char)); }

        if ( pixels != 0 )
        {
          size_t rd = fread(pixels,bytesPerPixel*channels, w*h, pf);
          if (rd < w*h )
             {
               fprintf(stderr,"Note : Incomplete read while reading file %s (%u instead of %u)\n",filename,(unsigned int) rd, w*h);
               fprintf(stderr,"Dimensions ( %u x %u ) , Depth %u bytes , Channels %u \n",w,h,bytesPerPixel,channels);
             }

          fclose(pf);

           #if PRINT_COMMENTS
             if ( (channels==1) && (bytesPerPixel==2) && (timestamp!=0) ) { printf("DEPTH %lu\n",*timestamp); } else
             if ( (channels==3) && (bytesPerPixel==1) && (timestamp!=0) ) { printf("COLOR %lu\n",*timestamp); }
           #endif

          return pixels;
        } else
        {
            fprintf(stderr,"Could not Allocate enough memory for file %s \n",filename);
        }
        fclose(pf);
    } else
    {
      fprintf(stderr,"File %s does not exist \n",filename);
    }


  //Compatibility with normal PNM files..!
  //swapEndianness(buffer,*width,*height,channels,bytesPerPixel*8);

  return buffer;
}
#endif




int makeFrameNoInput(unsigned char * frame , unsigned int width , unsigned int height , unsigned int channels)
{
   unsigned char * framePTR = frame;
   unsigned char * frameLimit = frame + width * height * channels * sizeof(char);

   while (framePTR<frameLimit)
   {
       *framePTR=0; ++framePTR;
       *framePTR=0; ++framePTR;
       *framePTR=255; ++framePTR;
   }
 return 1;
}



int flipDepth(unsigned short * depth,unsigned int width , unsigned int height )
{
  unsigned char tmp ;
  unsigned char * depthPtr=(unsigned char *) depth;
  unsigned char * depthPtrNext=(unsigned char *) depth+1;
  unsigned char * depthPtrLimit =(unsigned char *) depth + width * height * 2 ;
  while ( depthPtr < depthPtrLimit )
  {
     tmp=*depthPtr;
     *depthPtr=*depthPtrNext;
     *depthPtrNext=tmp;

     ++depthPtr;
     ++depthPtrNext;
  }

 return 0;
}


//Single place to change filename conventions :)
void getFilenameForNextResource(
                                char * filename ,
                                unsigned int maxSize ,
                                unsigned int resType ,
                                unsigned int devID ,
                                unsigned int cycle,
                                unsigned int * style,
                                char * readFromDir ,
                                char * colorSubDirectory ,
                                char * depthSubDirectory ,
                                char * extension
                                )
{
  devID=0; //<- test
  switch (resType)
  {
    case RESOURCE_COLOR_FILE :
        if (*style==1) { sprintf(filename,"frames/%s/%s/%06u.%s",readFromDir,colorSubDirectory,cycle,extension); } else
                       { sprintf(filename,"frames/%s/%s/colorFrame_%u_%05u.%s",readFromDir,colorSubDirectory,devID,cycle,extension); }
    break;
    case RESOURCE_DEPTH_FILE :
        if (*style==1) { sprintf(filename,"frames/%s/%s/%06u.%s",readFromDir,depthSubDirectory,cycle,extension); } else
                       { sprintf(filename,"frames/%s/%s/depthFrame_%u_%05u.%s",readFromDir,depthSubDirectory,devID,cycle,extension); }
    break;
    case RESOURCE_COLOR_CALIBRATION_FILE :  sprintf(filename,"frames/%s/color.calib",readFromDir); break;
    case RESOURCE_DEPTH_CALIBRATION_FILE :  sprintf(filename,"frames/%s/depth.calib",readFromDir); break;
    case RESOURCE_LIVE_CALIBRATION_FILE : sprintf(filename,"frames/%s/cameraPose_%u_%05u.calib",readFromDir,devID,cycle);
  };
}



unsigned int retreiveDatasetDeviceIDToReadFrom(
                                               unsigned int devID ,
                                               unsigned int cycle ,
                                               unsigned int * style,
                                               char * readFromDir ,
                                               char * colorSubDirectory ,
                                               char * depthSubDirectory ,
                                               char * extension
                                              )
{
 char * file_name_test = (char* ) malloc(MAX_DIR_PATH * sizeof(char));
 if (file_name_test==0) { fprintf(stderr,"Could not retreiveDatasetDeviceID , no space for string\n"); return 0; }

 unsigned int decided=0;
 unsigned int devIDInc=devID;
 while ( (devIDInc >=0 ) && (!decided) )
    {
      getFilenameForNextResource(file_name_test , MAX_DIR_PATH , RESOURCE_COLOR_FILE , devIDInc , cycle, style, readFromDir , colorSubDirectory, depthSubDirectory, extension );
      if (fileExists(file_name_test)) {  decided=1; }
      getFilenameForNextResource(file_name_test , MAX_DIR_PATH , RESOURCE_DEPTH_FILE , devIDInc , cycle, style, readFromDir , colorSubDirectory, depthSubDirectory, extension );
      if (fileExists(file_name_test)) {  decided=1; }

      if (devIDInc==0) { break; decided=1; } else
                       { --devIDInc; }
    }

  free(file_name_test);
  return devIDInc;
}


void * ReadImageFile(void * existingBuffer ,char * filename , char * extension ,  unsigned int * widthInternal, unsigned int * heightInternal, unsigned long *  timestampInternal)
{
 #if USE_CODEC_LIBRARY
   //Codec Library ignores existing buffers
   if (existingBuffer!=0) { free(existingBuffer); existingBuffer=0; }

   unsigned int type=0;
   if (strcmp(extension,"pnm")==0)  { type=PNM_CODEC; } else
   if (strcmp(extension,"cpnm")==0) { type=COMPATIBLE_PNM_CODEC; } else
   if (strcmp(extension,"png")==0)  { type=PNG_CODEC; } else
   if (strcmp(extension,"jpg")==0)  { type=JPG_CODEC; } else
   if (strcmp(extension,"jps")==0)  { type=JPG_CODEC; } else
   if (strcmp(extension,"jpeg")==0) { type=JPG_CODEC; }
   unsigned int bitsperpixel,channels;
   //fprintf(stderr,"Reading %s , using Codec Library .. ",filename);
   char * pixels = readImageRaw(filename,type,widthInternal,heightInternal,&bitsperpixel,&channels);
   //fprintf(stderr,"got back a %ux%u image \n",*widthInternal,*heightInternal);
   if ( (channels==1) && (bitsperpixel==16) && (type==PNG_CODEC) ) { swapImageEndiannessRaw( pixels, *widthInternal , *heightInternal ,bitsperpixel , channels);  }
   return pixels ;
 #else
   return ReadPNMInternal(existingBuffer,filename,widthInternal,heightInternal,timestampInternal);
 #endif // USE_CODEC_LIBRARY
}

unsigned int findSubdirsOfDataset(int devID, char * readFromDir , char * subColor , char * subDepth )
{
    strcpy(subColor,".");
    strcpy(subDepth,".");

    char folder[2048];
    snprintf(folder,2048,"frames/%s/color/",readFromDir);
    if (_tacq_directoryExists(folder)) { strcpy(subColor,"color"); }

    snprintf(folder,2048,"frames/%s/depth/",readFromDir);
    if (_tacq_directoryExists(folder)) { strcpy(subDepth,"depth"); }

  fprintf(stderr,"ColorSubDir : %s , DepthSubDir : %s \n",subColor,subDepth);
 return 1;
}


unsigned int findExtensionOfDataset(
                                     int devID,
                                     unsigned int * style ,
                                     char * readFromDir ,
                                     char * colorSubDirectory ,
                                     char * depthSubDirectory ,
                                     char * colorExtension ,
                                     char * depthExtension,
                                     unsigned int startingFrame
                                   )
{
  unsigned int colorSet=0,depthSet=0;
  char * file_name_test = (char* ) malloc(MAX_DIR_PATH * sizeof(char));
  if (file_name_test==0) { fprintf(stderr,"Could not findLastFrame , no space for string\n"); return 0; }

  unsigned int i=0,tryStyle=0;
  while (i<4)
  {
   if (i==0) { strncpy(colorExtension,"pnm",MAX_EXTENSION_PATH);  } else
   if (i==1) { strncpy(colorExtension,"png",MAX_EXTENSION_PATH);  } else
   if (i==2) { strncpy(colorExtension,"jpg",MAX_EXTENSION_PATH);  } else
   if (i==3) { strncpy(colorExtension,"jps",MAX_EXTENSION_PATH);  } else
   if (i==4) { strncpy(colorExtension,"cpnm",MAX_EXTENSION_PATH); }

   tryStyle=0;
   getFilenameForNextResource(file_name_test , MAX_DIR_PATH , RESOURCE_COLOR_FILE , devID , startingFrame , &tryStyle , readFromDir ,colorSubDirectory , depthSubDirectory , colorExtension );
   if ( fileExists(file_name_test) ) { *style=tryStyle; colorSet=1; break; }

   tryStyle=1;
   getFilenameForNextResource(file_name_test , MAX_DIR_PATH , RESOURCE_COLOR_FILE , devID , startingFrame , &tryStyle , readFromDir ,colorSubDirectory , depthSubDirectory , colorExtension );
   if ( fileExists(file_name_test) ) { *style=tryStyle; colorSet=1; break; }

   ++i;
  }

  i=0;
  while (i<4)
  {
   if (i==0) { strncpy(depthExtension,"pnm",MAX_EXTENSION_PATH);  } else
   if (i==1) { strncpy(depthExtension,"png",MAX_EXTENSION_PATH);  } else
   if (i==2) { strncpy(depthExtension,"jpg",MAX_EXTENSION_PATH);  } else
   if (i==3) { strncpy(depthExtension,"jps",MAX_EXTENSION_PATH);  } else
   if (i==4) { strncpy(colorExtension,"cpnm",MAX_EXTENSION_PATH); }

   tryStyle=0;
   getFilenameForNextResource(file_name_test , MAX_DIR_PATH , RESOURCE_DEPTH_FILE , devID , startingFrame , &tryStyle , readFromDir ,colorSubDirectory , depthSubDirectory , depthExtension );
   if ( fileExists(file_name_test) ) { *style=tryStyle; depthSet=1; break; }

   tryStyle=1;
   getFilenameForNextResource(file_name_test , MAX_DIR_PATH , RESOURCE_DEPTH_FILE , devID , startingFrame , &tryStyle , readFromDir ,colorSubDirectory , depthSubDirectory , depthExtension );
   if ( fileExists(file_name_test) ) { *style=tryStyle; depthSet=1; break; }

   ++i;
  }

  free(file_name_test);

  fprintf(stderr,"Color Extension : %s , Depth Extension : %s , Style : %u \n",colorExtension,depthExtension,*style);

  return ( (colorSet)&&(depthSet) );
}



unsigned int findLastFrame(
                           int devID,
                           unsigned int * style ,
                           char * readFromDir ,
                           char * colorSubDirectory ,
                           char * depthSubDirectory ,
                           char * colorExtension,
                           char * depthExtension
                           )
{
  unsigned int totalFrames=0;
  unsigned int i=0;

  char * file_name_test = (char* ) malloc(MAX_DIR_PATH * sizeof(char));
  if (file_name_test==0) { fprintf(stderr,"Could not findLastFrame , no space for string\n"); return 0; }

  //TODO : This can be done in a much smarter way , this is the most stupid way possible to do this kind of check :P
  while (i<100000)
  {
   totalFrames = i;
   getFilenameForNextResource(file_name_test , MAX_DIR_PATH , RESOURCE_COLOR_FILE , devID , i , style , readFromDir ,colorSubDirectory , depthSubDirectory , colorExtension );
   if ( ! fileExists(file_name_test) ) { break; }
   getFilenameForNextResource(file_name_test , MAX_DIR_PATH , RESOURCE_DEPTH_FILE , devID , i , style , readFromDir ,colorSubDirectory , depthSubDirectory , depthExtension );
   if ( ! fileExists(file_name_test) ) { break; }
   ++i;
  }

  free(file_name_test);

  return totalFrames;
}

