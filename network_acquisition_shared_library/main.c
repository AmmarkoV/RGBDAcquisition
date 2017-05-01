#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "NetworkAcquisition.h"


#include "../tools/Codecs/codecs.h"
#include "../tools/Codecs/jpgInput.h"

#include "../acquisition/Acquisition.h"

#include "daemon.h"


#include <string.h>
#include <math.h>


#define PRINT_COMMENTS 1
#define PRINT_DEBUG_EACH_CALL 0

#define DEFAULT_FOCAL_LENGTH 120.0
#define DEFAULT_PIXEL_SIZE 0.1052

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */

#define BUILD_NETWORK 1

#if BUILD_NETWORK


 FILE *output;

struct NetworkVirtualDevice networkDevice[MAX_NETWORK_DEVICES]={0};

/*
   This library is unique in the sense that instead of only having the regular calls all the other plugin acquisition modules have
   it has the following three calls that allow it to "broadcast" and not only "receive" input from an aquisition source..
*/


//This call creates a FrameServer that binds to the IP and port given and from there on starts to
//service the clients that connect to it by sending frames to them..
int networkBackbone_startPushingToRemoteOLD(char * ip , int port)
{
  fprintf(stderr,"networkBackbone_startPushingToRemote(%s,%u) called \n",ip,port);
  return StartFrameServer(0,ip,port);
}

int networkBackbone_stopPushingToRemoteOLD(int frameServerID)
{
  return StopFrameServer(frameServerID);
}

//This call pushes an image to be made available to the clients connected to our frameserver

int networkBackbone_pushImageToRemoteOLD(int frameServerID, int streamNumber , void* pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel)
{
  if (streamNumber==0) //Color
  {
      networkDevice[0].okToSendColorFrame=0;
      networkDevice[0].colorWidth=width;
      networkDevice[0].colorHeight=height;
      networkDevice[0].colorChannels=channels;
      networkDevice[0].colorBitsperpixel=bitsperpixel;
      networkDevice[0].colorFrame = (unsigned char*) pixels;
      networkDevice[0].compressedColorSize=0; // Not using compression

      /*
      struct Image * img = createImageUsingExistingBuffer(width,height,channels,bitsperpixel,pixels);
      networkDevice[0].compressedColorSize=64*1024; //64KBmax
      char * compressedPixels = (char* ) malloc(sizeof(char) * networkDevice[0].compressedColorSize);
      WriteJPEGInternal("dummyName.jpg",img,compressedPixels,&networkDevice[0].compressedColorSize);
      networkDevice[0].colorFrame = (char*) compressedPixels;
      fprintf(stderr,"Compressed from %u bytes\n",networkDevice[0].compressedColorSize);*/


      networkDevice[0].okToSendColorFrame=1;


      while (networkDevice[0].okToSendColorFrame==1)
       {
         usleep(1000);
         fprintf(stderr,"Cf.");
       }
  }
   else
 if (streamNumber==1) //Depth
  {
      networkDevice[0].okToSendDepthFrame=0;
      networkDevice[0].depthWidth=width;
      networkDevice[0].depthHeight=height;
      networkDevice[0].depthChannels=channels;
      networkDevice[0].depthBitsperpixel=bitsperpixel;
      networkDevice[0].depthFrame = (unsigned short*) pixels;

      networkDevice[0].okToSendDepthFrame=1;


      while (networkDevice[0].okToSendDepthFrame==1)
       {
         usleep(1000);
         fprintf(stderr,"Df.");
       }
  }

  return 0;
}



int networkBackbone_startPushingToRemote(char * ip , int port , unsigned int width , unsigned int height)
{
  //ffserver
  //ffmpeg -i dota_stripped.avi -f matroska - | ffplay -
  //ffmpeg -i dota_stripped.avi -i http://localhost:8090/feed1.ffm
  //ffplay http://localhost:8090/feed1.ffm


   //ffmpeg -y -f rawvideo -vcodec rawvideo -s 420x360 -pix_fmt rgb24 -r 24 -i - -an -vcodec mpeg my_output_videofile.mp4

  //DUMP TO FILE DEBUG
  //output = popen ("ffmpeg -y -f rawvideo -vcodec rawvideo -s 640x480 -pix_fmt rgb24 -r 24 -i - -f mp4 -q:v 5 -an -vcodec mpeg4 test.mp4", "w");

  //output = popen ("ffmpeg -y -f lavfi -i aevalsrc=0  -f rawvideo -vcodec rawvideo -s 640x480 -pix_fmt rgb24 -r 24 -i - -acodec copy -b:a 32k http://localhost:8090/feed1.ffm", "w");
  //-map 1:0
  //output = popen ("ffmpeg -y -f lavfi -i aevalsrc=0 -c:a pcm_s16le  -f rawvideo -vcodec rawvideo -s 640x480 -pix_fmt rgb24 -r 24  -i - -vf \"format=yuv420p\" -g 60 -f flv http://localhost:8090/feed1.ffm", "w");
  output = popen ("ffmpeg -y -f rawvideo -vcodec rawvideo -s 640x480 -pix_fmt rgb24 -r 24  -i - -vf \"format=yuv420p\" -g 60 -f flv rtmp://a.rtmp.youtube.com/live2/", "w");
  if (!output)
    {
      fprintf (stderr,"incorrect parameters or too many files.\n");
      return 0;
    }
  return 1;
}

int networkBackbone_stopPushingToRemote(int frameServerID)
{
  if (pclose (output) != 0)
    {
      fprintf (stderr,"Could not run more or other error.\n");
      return 0;
    }
  return 1;
}




int networkBackbone_pushImageToRemote(int frameServerID, int streamNumber , void* pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel)
{
  fprintf(stderr,"networkBackbone_pushImageToRemote\n");
  if (streamNumber==0) //Color
  {

      /*
      struct Image * img = createImageUsingExistingBuffer(width,height,channels,bitsperpixel,pixels);
      if (img!=0)
      {
       unsigned long compressedColorSize=64*1024; //64KBmax
       char * compressedPixels = (char* ) malloc(sizeof(char) * compressedColorSize);
       if (compressedPixels!=0)
        {
         WriteJPEGMemory(img,compressedPixels,&compressedColorSize);
         fprintf(stderr,"Compressed from %lu bytes\n",compressedColorSize);
         fwrite (compressedPixels, sizeof(char), compressedColorSize, output);
        }
      }
      */
     fwrite (pixels, sizeof(char), width*height*channels, output);

  }

  return 0;
}

















double getNetworkColorPixelSize(int devID)   { return DEFAULT_PIXEL_SIZE; }
double getNetworkColorFocalLength(int devID)
{
   return DEFAULT_FOCAL_LENGTH;
}

double getNetworkDepthFocalLength(int devID) {  return DEFAULT_FOCAL_LENGTH; }
double getNetworkDepthPixelSize(int devID) { return DEFAULT_PIXEL_SIZE; }



int getNetworkColorCalibration(int devID,struct calibration * calib) { return 1; }
int getNetworkDepthCalibration(int devID,struct calibration * calib) { return 1; }
int setNetworkColorCalibration(int devID,struct calibration * calib) { return 1; }
int setNetworkDepthCalibration(int devID,struct calibration * calib) {     return 1; }





int startNetworkModule(unsigned int max_devs,char * settings)
{
    fprintf(stderr,"\n\n\n\n\n\n");
    fprintf(stderr,RED "The library that you are trying to use is not implemented (YET) ;P !\n" NORMAL );
    fprintf(stderr,RED "Please note that most of the Network module is a stub\n" NORMAL );
    fprintf(stderr,RED "If you really want it consider commenting or requesting it on the github repository https://github.com/AmmarkoV/RGBDAcquisition/issues/2 !\n" NORMAL );
    fprintf(stderr,RED "For now and unless there is interest from someone else this is a low priority !\n" NORMAL );

    fprintf(stderr,"\n\n\n\n\n\n");
    return 0;
}




int getNetworkNumberOfDevices() { return 1; }

int stopNetworkModule() {    return 1; }

int createNetworkDevice(int devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate) { return 0; }
int destroyNetworkDevice(int devID) { return 1; }
int seekNetworkFrame(int devID,unsigned int seekFrame) { return 1; }
int snapNetworkFrames(int devID) { return 1; }

//Color Frame getters
unsigned long getLastNetworkColorTimestamp(int devID) { return networkDevice[devID].lastColorTimestamp; }
int getNetworkColorWidth(int devID)        { return networkDevice[devID].colorWidth; }
int getNetworkColorHeight(int devID)       { return networkDevice[devID].colorHeight; }
int getNetworkColorDataSize(int devID)     { return networkDevice[devID].colorWidth * networkDevice[devID].colorHeight * 3; }
int getNetworkColorChannels(int devID)     { return 3; }
int getNetworkColorBitsPerPixel(int devID) { return 8; }

// Frame Grabber should call this function for color frames
unsigned char * getNetworkColorPixels(int devID)    { return networkDevice[devID].colorFrame; }




   //Depth Frame getters
unsigned long getLastNetworkDepthTimestamp(int devID) { return networkDevice[devID].lastDepthTimestamp; }
int getNetworkDepthWidth(int devID)    { return networkDevice[devID].depthWidth; }
int getNetworkDepthHeight(int devID)   { return networkDevice[devID].depthHeight; }
int getNetworkDepthDataSize(int devID) { return networkDevice[devID].depthWidth * networkDevice[devID].depthHeight; }
int getNetworkDepthChannels(int devID)     { return 1; }
int getNetworkDepthBitsPerPixel(int devID) { return 16; }

// Frame Grabber should call this function for depth frames
unsigned char * getNetworkDepthPixels(int devID) { return (unsigned char *) networkDevice[devID].depthFrame; }

#else
//Null build
int startNetworkModule(unsigned int max_devs,char * settings)
{
    fprintf(stderr,"startNetworkModule called on a dummy build of NetworkAcquisition!\n");
    fprintf(stderr,"Please consider enabling #define BUILD_NETWORK 1 on acquisition/acquisition_setup.h\n");
    return 0;
  return 1;
}
#endif
