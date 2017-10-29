
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "PixelFormatConversions.h"
#include "V4L2Wrapper.h"

char video_simulation_path[256]={0};


int total_cameras=0;
unsigned char * empty_frame=0;
unsigned int largest_feed_x=640;
unsigned int largest_feed_y=480;

struct Video * camera_feeds=0;


io_method io=IO_METHOD_MMAP;//  /*IO_METHOD_MMAP;  IO_METHOD_READ; IO_METHOD_USERPTR;*/



int DecodePixels(int webcam_id)
{
 if ( camera_feeds[webcam_id].frame_decoded==0)
     { /*THIS FRAME HASN`T BEEN DECODED YET!*/
       int i=Convert2RGB24( (unsigned char*)camera_feeds[webcam_id].frame,
                            (unsigned char*)camera_feeds[webcam_id].decoded_pixels,
                            camera_feeds[webcam_id].width,
                            camera_feeds[webcam_id].height,
                            camera_feeds[webcam_id].input_pixel_format,
                            camera_feeds[webcam_id].input_pixel_format_bitdepth );
       if ( i == 0 ) { /* UNABLE TO PERFORM CONVERSION */ return 0; } else
                     { /* SUCCESSFUL CONVERSION */
                       camera_feeds[webcam_id].frame_decoded=1;
                     }
    }
 return 1;
}


unsigned char * ReturnDecodedLiveFrame(int webcam_id)
{
   /*
          THIS FRAME DECIDES IF THE VIDEO FORMAT NEEDS DECODING OR CAN BE RETURNED RAW FROM THE DEVICE
          SEE PixelFormats.cpp / PixelFormatConversions.cpp
   */

   if (VideoFormatNeedsDecoding(camera_feeds[webcam_id].input_pixel_format,camera_feeds[webcam_id].input_pixel_format_bitdepth)==1)
    {
     /*VIDEO COMES IN A FORMAT THAT NEEDS DECODING TO RGB 24*/
     if ( DecodePixels(webcam_id)==0 ) return empty_frame;
     return (unsigned char *) camera_feeds[webcam_id].decoded_pixels;
    } else
    {
      /* The frame is ready so we mark it as decoded*/
      camera_feeds[webcam_id].frame_decoded=1;
      if ( camera_feeds[webcam_id].frame == 0 )
         {
           /*Handler for when the frame does not exist */
           return empty_frame;
         }
     return (unsigned char *) camera_feeds[webcam_id].frame;
    }
   return empty_frame;
}



int VideoInput_InitializeLibrary(int numofinputs)
{
    //if (total_cameras>0) { fprintf(stderr,"Error , Video Inputs already active ?\n total_cameras=%u\n",total_cameras); return 0;}

    //ReallocEmptyFrame(largest_feed_x,largest_feed_y);


    /*First allocate memory for V4L2 Structures  , etc*/
    camera_feeds = (struct Video * ) malloc ( sizeof( struct Video ) * (numofinputs+1) );
    if (camera_feeds==0) { fprintf(stderr,"Error , cannot allocate memory for %u video inputs \n",total_cameras); return 0;}

    int i;
    for ( i=0; i<total_cameras; i++ )
      {  /*We mark each camera as dead , to preserve a clean state*/
          camera_feeds[i].loop_thread = 0;

          camera_feeds[i].thread_alive_flag=0;
         // camera_feeds[i].rec_video.pixels=0;
          camera_feeds[i].frame=0;
          //camera_feeds[i].v4l2_intf=0;
          memset(&camera_feeds[i].v4l2_interface , 0 , sizeof (struct V4L2_c_interface)) ;
          camera_feeds[i].fx=0,camera_feeds[i].fy=0,camera_feeds[i].cx=0,camera_feeds[i].cy=0;
          camera_feeds[i].k1=0,camera_feeds[i].k2=0,camera_feeds[i].p1=0,camera_feeds[i].p2=0,camera_feeds[i].k3=0;

          //memset ((void*) camera_feeds[i],0,sizeof(struct Video));
      }

    /*Lets Refresh USB devices list :)*/
    if (VIDEOINPUT_DEBUG)
    {
      int ret=system((const char * ) "ls /dev/video*");
      if ( ret == 0 ) { printf("These are the possible video devices .. \n"); }

      ret=system((const char * ) "ls /dev/video* | wc -l");
      if ( ret == 0 ) { printf("total video devices .. \n"); }
    }



    total_cameras=numofinputs;

    return 1 ;
}

int VideoInput_DeinitializeLibrary()
{
    if (total_cameras==0) { fprintf(stderr,"Error , Video Inputs already deactivated ?\n"); return 0;}
    if (camera_feeds==0) { fprintf(stderr,"Error , Video Inputs already deactivated ?\n"); return 0;}

    if ( empty_frame != 0 ) { free(empty_frame); empty_frame=0; }

    int i=0;
    for ( i=0; i<total_cameras; i++ )
     {
       if (camera_feeds[i].thread_alive_flag)
       {
        fprintf(stderr,"Video %u Stopping\n",i);
        camera_feeds[i].stop_snap_loop=1;

        //if ( pthread_join( camera_feeds[i].loop_thread, NULL) != 0 )           { fprintf(stderr,"Error rejoining VideoInput thread \n"); }

        usleep(30);
        //camera_feeds[i].v4l2_intf->stopCapture();
        stopCapture_v4l2intf(&camera_feeds[i].v4l2_interface);
        usleep(30);
        //camera_feeds[i].v4l2_intf->freeBuffers();
        freeBuffers_v4l2intf(&camera_feeds[i].v4l2_interface);
        usleep(30);

        if ( camera_feeds[i].decoded_pixels !=0 )   { free( camera_feeds[i].decoded_pixels );   }
      //  if ( camera_feeds[i].rec_video.pixels !=0 ) { free( camera_feeds[i].rec_video.pixels ); }
        //if ( camera_feeds[i].v4l2_intf != 0 )       { delete camera_feeds[i].v4l2_intf; }
        destroy_v4l2intf(&camera_feeds[i].v4l2_interface);

       } else
       {
        fprintf(stderr,"Video Feed %u seems to be already dead , ensuring no memory leaks!\n",i);
        camera_feeds[i].stop_snap_loop=1;
        //if ( camera_feeds[i].rec_video.pixels !=0 ) { free( camera_feeds[i].rec_video.pixels ); }
        //if ( camera_feeds[i].v4l2_intf != 0 )       { delete camera_feeds[i].v4l2_intf; }
        destroy_v4l2intf(&camera_feeds[i].v4l2_interface);

       }
     }



    fprintf(stderr,"Deallocation of Video Structures\n");
    free(camera_feeds);

    fprintf(stderr,"Video Input successfully deallocated\n");
    return 1 ;
}



char FileExistsVideoInput(const char * filename)
{
 FILE *fp = fopen(filename,"r");
 if( fp ) { /* exists */
            fclose(fp);
            return 1;
          }
          else
          { /* doesnt exist */ }
 return 0;
}

int VideoInput_OpenFeed(int inpt,const char * viddev,int width,int height,int bitdepth,int framespersecond,char snapshots_on,struct VideoFeedSettings videosettings)
{
   camera_feeds[inpt].video_simulation=NO_VIDEO_AVAILIABLE;
   printf("Initializing Video Feed %u ( %s ) @ %u/%u \n",inpt,viddev,width,height);
   //ReallocEmptyFrame(width,height);

   //if (!VideoInputsOk()) return 0;
   if ( (!FileExistsVideoInput(viddev)) ) { fprintf(stderr,"\n\nCheck for the webcam (%s) returned false..\n PLEASE CONNECT V4L2 COMPATIBLE CAMERA!!!!!\n\n\n",viddev); return 0; }


   camera_feeds[inpt].videoinp = viddev; /*i.e. (char *) "/dev/video0";*/
   camera_feeds[inpt].width = width;
   camera_feeds[inpt].height = height;
   camera_feeds[inpt].size_of_frame=width*height*(bitdepth/8);
   //We will set this at the end..! camera_feeds[inpt].video_simulation=LIVE_ON;
   camera_feeds[inpt].thread_alive_flag=0;
   camera_feeds[inpt].snap_paused=0;
   camera_feeds[inpt].snap_lock=0;

   camera_feeds[inpt].frame_decoded=0;
   camera_feeds[inpt].decoded_pixels=0;


   CLEAR (camera_feeds[inpt].fmt);
   camera_feeds[inpt].fmt.fmt.pix.width       = width;
   camera_feeds[inpt].fmt.fmt.pix.height      = height;


   /* IF videosettings is null set default capture mode ( VIDEO CAPTURE , YUYV mode , INTERLACED )  */
   if ( videosettings.EncodingType==0 ) { camera_feeds[inpt].fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; } else
                                        { camera_feeds[inpt].fmt.type =  videosettings.EncodingType; }

   if ( videosettings.PixelFormat==0 ) { camera_feeds[inpt].fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV; } else
                                       { camera_feeds[inpt].fmt.fmt.pix.pixelformat = videosettings.PixelFormat; }

   if ( videosettings.FieldType==0 ) { camera_feeds[inpt].fmt.fmt.pix.field = V4L2_FIELD_NONE; /*V4L2_FIELD_INTERLACED;*/ } else
                                     { camera_feeds[inpt].fmt.fmt.pix.field =  videosettings.FieldType; }

   camera_feeds[inpt].input_pixel_format=camera_feeds[inpt].fmt.fmt.pix.pixelformat;
   camera_feeds[inpt].input_pixel_format_bitdepth=bitdepth;

   PrintOutCaptureMode(camera_feeds[inpt].fmt.type);
   PrintOutPixelFormat(camera_feeds[inpt].fmt.fmt.pix.pixelformat);
   PrintOutFieldType(camera_feeds[inpt].fmt.fmt.pix.field);

   camera_feeds[inpt].decoded_pixels=0;


       if (!VideoFormatImplemented(camera_feeds[inpt].input_pixel_format,camera_feeds[inpt].input_pixel_format_bitdepth))
       {
          fprintf(stderr,(char *)"Video format not implemented!!! :S \n");
       }

       if (VideoFormatNeedsDecoding(camera_feeds[inpt].input_pixel_format,camera_feeds[inpt].input_pixel_format_bitdepth))
       {
          /*NEEDS TO DECODE TO RGB 24 , allocate memory*/
          camera_feeds[inpt].decoded_pixels = (char * ) malloc( (width*height*3) + 1);
          if (camera_feeds[inpt].decoded_pixels ==0) { fprintf(stderr,"Error allocating memory for DECODER TO RGB 24 structure"); return 0; }
          memset(camera_feeds[inpt].decoded_pixels, '\0',width*height*3);
       }



      fprintf(stderr,(char *)"Starting camera , if it segfaults consider running \nLD_PRELOAD=/usr/lib/libv4l/v4l2convert.so  executable_name\n");


      if (!populateAndStart_v4l2intf(&camera_feeds[inpt].v4l2_interface,camera_feeds[inpt].videoinp,io) ) { fprintf(stderr,"Could not populate and start v4l2 interface..\n"); return 0; }

       struct v4l2_capability cap;
       if (getcap_v4l2intf(&camera_feeds[inpt].v4l2_interface,&cap))
       {
           fprintf(stderr,"Device Bus ....... %s \n" ,cap.bus_info);
           fprintf(stderr,"Card       ....... %s \n" ,cap.card);
           fprintf(stderr,"Driver     ....... %s \n" ,cap.driver);
           fprintf(stderr,"Driver Version ... %d.%d.%d\n",cap.version >> 16,(cap.version >> 8) & 0xff,cap.version & 0xff);
		   fprintf(stderr,"Capabilities ..... 0x%08X\n", cap.capabilities);
		   fprintf(stderr,"DevCapabilities .. 0x%08X\n", cap.device_caps);

		   //print_video_formats_ext(&camera_feeds[inpt].v4l2_interface,V4L2_BUF_TYPE_VIDEO_CAPTURE);
		   /*
		   fprintf(stderr,"%s", cap2s(cap.capabilities).c_str());
		   if (vcap.capabilities & V4L2_CAP_DEVICE_CAPS)
              {
			   printf("\tDevice Caps   : 0x%08X\n", vcap.device_caps);
			   printf("%s", cap2s(vcap.device_caps).c_str());
		      }*/

           if ( ! cap.device_caps & V4L2_CAP_TIMEPERFRAME )
              { fprintf(stderr,"This Device has no framerate changing capabilities ?\n"); framespersecond = 0; } else
              {
                //struct v4l2_frmivalenum argp;
                //getFramerateIntervals_v4l2intf(&camera_feeds[inpt].v4l2_interface,&argp);
                //fprintf(stderr,"Todo check if our resolution is supported here\n");
              }
       }

      if ( setfmt_v4l2intf(&camera_feeds[inpt].v4l2_interface,camera_feeds[inpt].fmt) == 0 )
                { fprintf(stderr,"Device does not support settings:\n"); return 0; } else
                 {
                    if (VIDEOINPUT_DEBUG) { fprintf(stderr,"No errors , starting camera %u / locking memory..!",inpt); }

                    //This doesnt work as expected
                    if (framespersecond!=0)
                      {
                          if (!setFramerate_v4l2intf(&camera_feeds[inpt].v4l2_interface , (unsigned int) framespersecond ) )
                                              { fprintf(stderr,"Could not set Framerate to %d  ..\n" ,framespersecond);  } else
                                              { fprintf(stderr,"Set Framerate to %d  ..\n" ,framespersecond);   }
                      }

                    if ( !initBuffers_v4l2intf(&camera_feeds[inpt].v4l2_interface) ) { fprintf(stderr,"Could not initialize buffers..\n"); return 0; }
                    if ( !startCapture_v4l2intf(&camera_feeds[inpt].v4l2_interface) ) { fprintf(stderr,"Could not start capture..\n"); return 0; }

                    camera_feeds[inpt].frame = empty_frame;
                 }


    /* INIT MEMORY FOR SNAPSHOTS !*/


    /* STARTING VIDEO RECEIVE THREAD!*/
    camera_feeds[inpt].thread_alive_flag=0; /* <- This will be set to 1 when child process will start :)*/
    camera_feeds[inpt].stop_snap_loop=0;
    camera_feeds[inpt].loop_thread=0;
    camera_feeds[inpt].frameNumber=0;

   // ChooseDifferentSoftFramerate(inpt,framespersecond); // Go for a good old solid PAL 25 fps , ( the PS3 cameras may be snapping at 120fps , but VisualCortex without
                                           // hardware acceleration can`t go more than 6-8 fps )



    /*
    unsigned int waittime=0,MAX_WAIT=10  , SLEEP_PER_LOOP_MILLI = 20 *  1000;// Milliseconds
    waittime = MAX_WAIT; // <- disable this
    if (VIDEOINPUT_DEBUG) { printf("Giving some time ( max =  %u ms ) for the receive threads to wake up ",MAX_WAIT*SLEEP_PER_LOOP_MILLI); }
    while ( ( waittime<MAX_WAIT ) && (camera_feeds[inpt].thread_alive_flag==0) ) {
                                                                                   if (waittime%10==0) printf(".");
                                                                                   usleep(SLEEP_PER_LOOP_MILLI);
                                                                                   ++waittime;
                                                                                 }
   */

    if (VIDEOINPUT_DEBUG) { printf("\nInitVideoFeed %u is ok!\n",inpt); }

    camera_feeds[inpt].video_simulation=LIVE_ON;

  return 1;
}

