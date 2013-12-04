#ifndef V4L2WRAPPER_H_INCLUDED
#define V4L2WRAPPER_H_INCLUDED


#include "V4L2_c.h"

#include <linux/types.h>
#include <linux/videodev2.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include <errno.h>

#include <asm/types.h>          /* for videodev2.h */

#define MAX_DEVICE_FILENAME 256

#define TIMEOUT_SEC 10
#define TIMEOUT_USEC 0

#define DO_NOT_RETURN_NULL_POINTERS 1
#define VIDEOINPUT_DEBUG 0
#define VIDEOINPUT_INCREASEPRIORITY 0


enum input_modes
{
   LIVE_ON=0,
   RECORDING_ON,
   RECORDING_ONE_ON,
   PLAYBACK_ON,
   PLAYBACK_ON_LOADED,
   WORKING,
   NO_VIDEO_AVAILIABLE
};



struct VideoFeedSettings
{
   unsigned int EncodingType;
   unsigned int PixelFormat; // YOU CAN CHOOSE ONE OF THE FOLLOWING
   unsigned int FieldType;
};



struct Video
{
  /* DEVICE NAME */
  char * videoinp;
  unsigned int height;
  unsigned int width;
  unsigned int frame_rate;
  unsigned int sleep_time_per_frame_microseconds;
  unsigned int frame_already_passed;

  /* VIDEO 4 LINUX DATA */
  struct v4l2_format fmt;
  void *frame;
  unsigned int size_of_frame;
  //V4L2 *v4l2_intf;
  struct V4L2_c_interface v4l2_interface;

  int enableIntrinsicResectioning;
  unsigned int * resectionPrecalculations;
  /* CAMERA INTRINSIC PARAMETERS */
  float fx,fy,cx,cy;
  float k1,k2,p1,p2,k3;

  /* CAMERA EXTRINSIC PARAMETERS */
  float extrinsicR[9];
  float extrinsicT[3];


  /* DATA NEEDED FOR DECODERS TO WORK */
  unsigned int input_pixel_format;
  unsigned int input_pixel_format_bitdepth;
  char * decoded_pixels;
  int frame_decoded;

  /*VIDEO SIMULATION DATA*/
  //struct Image rec_video;
  int video_simulation;
  int keep_timestamp;
  int compress;

  int jpeg_compressor_running;
  char * mem_buffer_for_recording;
  unsigned long mem_buffer_for_recording_size;

  /* THREADING DATA */
  int thread_alive_flag;
  int snap_paused; /* If set to 1 Will continue to snap frames but not save the reference ( that way video loop wont die out ) */
  int snap_lock; /* If set to 1 Will not snap frames at all ( video loop will die out after a while ) */
  int stop_snap_loop;
  pthread_t loop_thread;
};


extern char video_simulation_path[256];

extern int total_cameras;
extern unsigned char * empty_frame;
extern unsigned int largest_feed_x;
extern unsigned int largest_feed_y;

extern struct Video * camera_feeds;

unsigned char * ReturnDecodedLiveFrame(int webcam_id);

int VideoInput_InitializeLibrary(int numofinputs);
int VideoInput_DeinitializeLibrary();

int VideoInput_OpenFeed(int inpt,char * viddev,int width,int height,int bitdepth,int framespersecond,char snapshots_on,struct VideoFeedSettings videosettings);


#endif // V4L2WRAPPER_H_INCLUDED
