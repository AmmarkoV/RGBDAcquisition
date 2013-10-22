#include "FeedScreenMemory.h"
#include <wx/wx.h>
#include <wx/dc.h>
#include <wx/utils.h>

wxMouseState mouse;
wxBitmap *default_feed=0;
wxBitmap *default_patch=0;

struct feed live_feeds[4]={{0},{0},{0},{0}};


unsigned int last_viscrtx_pass=0;
unsigned int resc_width=640,resc_height=480;
int VideoFeedsNotAccessible = 0;

int  has_init=1;

void memcpy_1bit_2_3bit(unsigned char * dest,unsigned char * src,unsigned int length)
{
  register unsigned char  *px;
  register unsigned char  *r;
  register unsigned char  *g;
  register unsigned char  *b;
  unsigned char val;
  px = (unsigned char  *)  dest;

  for ( unsigned int ptr = 0; ptr < length; ptr++)
   {
       r = px++;
       g = px++;
       b = px++;

       val = ( unsigned char ) src[ptr];
       *r= val;
       *g= val;
       *b= val;
   }
}


void PassVideoRegisterToFeed(unsigned int feednum,void * framedata,unsigned int bitsperpixel)
{
  void *frame=0;

  if ( live_feeds[feednum].bmp == 0 ) {  return;  } else
  if ( live_feeds[feednum].bmp_allocated ) { delete live_feeds[feednum].bmp; live_feeds[feednum].bmp_allocated = false; }
  frame = framedata;

 if ( frame != 0)
 {
    live_feeds[feednum].img.SetData((unsigned char *)framedata,live_feeds[feednum].width,live_feeds[feednum].height,true);
    if ( resc_width != live_feeds[feednum].width ) { live_feeds[feednum].img.Rescale(resc_width,resc_height); }
    live_feeds[feednum].bmp= new wxBitmap(live_feeds[feednum].img);
    live_feeds[feednum].bmp_allocated = true;
 }
}


void initFeeds()
{
    wxString filename;

    filename.Clear();
    //filename<<wxString(GetRoboKernelEnvPath(),wxConvUTF8);
    filename<<wxT("default.bmp");
    default_feed=new wxBitmap(filename,wxBITMAP_TYPE_BMP);


    int i=0;
    for ( i=0; i<4; i++)
    {
      live_feeds[i].bmp=default_feed;
      live_feeds[i].bmp_allocated = false;
      live_feeds[i].frame = malloc( live_feeds[i].width * live_feeds[i].height * 3 * sizeof (unsigned char) );
      if (live_feeds[i].frame==0)
      {
          fprintf(stderr,"Error allocating memory for feed %u \n",i);
      }
    }

    VideoFeedsNotAccessible=1;
    has_init=1;
}

void closeFeeds()
{
    if ( has_init == 0 ) { return; }
    has_init=0;

    VideoFeedsNotAccessible=0;

    delete default_feed;
    delete default_patch;

    int i=0;
    for ( i=0; i<4; i++)
    {
      if (live_feeds[i].bmp_allocated) { delete live_feeds[i].bmp; live_feeds[i].bmp_allocated = false; }
      if ( live_feeds[i].frame != 0 ) free( live_feeds[i].frame );
    }

     fprintf(stderr,"Gracefull exit :) \n");
}



bool XYOverFeed(int x , int y , feed feedmem)
{
   if ( (x>=feedmem.x1) && (x<=feedmem.x2) )
     {
         if ( (y>=feedmem.y1) && (y<=feedmem.y2) )
         {
           return true;
         }
     }
   return false;
}

void CheckMousePosition()
{
    mouse = wxGetMouseState();
    int x=mouse.GetX();
    int y=mouse.GetY();

    if ( XYOverFeed(x,y,live_feeds[2]) ) {}  else
    if ( XYOverFeed(x,y,live_feeds[3]) ) {}
}


int SaveRegisterToFile(char * filename , unsigned int reg_num)
{
//   return VisCortX_SaveVideoRegisterToFile(reg_num,filename);
}

int SnapWebCams()
{

 if (has_init==0) { fprintf(stderr,"!"); VideoFeedsNotAccessible=1; return 0; }

/*
 if ( (!VisCortx_VideoRegistersSynced(LEFT_EYE,CALIBRATED_LEFT_EYE)) ||
      (!VisCortx_VideoRegistersSynced(RIGHT_EYE,CALIBRATED_RIGHT_EYE))
     )
  {
       fprintf(stderr,"Feeds not synchronized , not copying frame \n");
       return 0;
  }

 if (last_viscrtx_pass==VisCortx_GetTime()) { return 0; }
 last_viscrtx_pass=VisCortx_GetTime();*/

 VideoFeedsNotAccessible=0;

 if ( live_feeds[0].bmp_allocated ) { live_feeds[0].bmp_allocated = false; delete live_feeds[0].bmp;  live_feeds[0].bmp=0; }
 if ( live_feeds[1].bmp_allocated ) { live_feeds[1].bmp_allocated = false; delete live_feeds[1].bmp;  live_feeds[1].bmp=0; }

/*
 if ( GetVideoRegister(CALIBRATED_LEFT_EYE) != 0)
 {                                                //
    live_feeds[0].img.SetData(GetVideoRegister(CALIBRATED_LEFT_EYE),GetCortexMetric(RESOLUTION_X),GetCortexMetric(RESOLUTION_Y),true);
    if ( resc_width != GetCortexMetric(RESOLUTION_X) ) { live_feeds[0].img.Rescale(resc_width,resc_height); }
    live_feeds[0].bmp= new wxBitmap(live_feeds[0].img);
    live_feeds[0].bmp_allocated = true;
 }

 if ( GetVideoRegister(CALIBRATED_RIGHT_EYE) != 0)
 {                                               //
  live_feeds[1].img.SetData(GetVideoRegister(CALIBRATED_RIGHT_EYE),GetCortexMetric(RESOLUTION_X),GetCortexMetric(RESOLUTION_Y),true);
  if ( resc_width != GetCortexMetric(RESOLUTION_X) ) { live_feeds[1].img.Rescale(resc_width,resc_height); }
  live_feeds[1].bmp= new wxBitmap(live_feeds[1].img);
  live_feeds[1].bmp_allocated = true;
 }



if ( last_time_of_left_operation != VisCortx_GetVideoRegisterData(LAST_LEFT_OPERATION,0) )
    {
         last_time_of_left_operation= VisCortx_GetVideoRegisterData(LAST_LEFT_OPERATION,0);
         PassVideoRegisterToFeed ( 2 , VisCortx_ReadFromVideoRegister(LAST_LEFT_OPERATION,GetCortexMetric(RESOLUTION_X),GetCortexMetric(RESOLUTION_Y),3),3 );
    } else
  if ( last_time_of_left_depth_map != VisCortx_GetVideoRegisterData(DEPTH_LEFT_VIDEO,0) )
    {
         last_time_of_left_depth_map= VisCortx_GetVideoRegisterData(DEPTH_LEFT_VIDEO,0);
         PassVideoRegisterToFeed ( 2 , VisCortx_ReadFromVideoRegister(DEPTH_LEFT_VIDEO,GetCortexMetric(RESOLUTION_X),GetCortexMetric(RESOLUTION_Y),3),3 );
    }


  if ( last_time_of_right_operation != VisCortx_GetVideoRegisterData(LAST_RIGHT_OPERATION,0) )
    {
         last_time_of_right_operation= VisCortx_GetVideoRegisterData(LAST_RIGHT_OPERATION,0);
         PassVideoRegisterToFeed ( 3 , VisCortx_ReadFromVideoRegister(LAST_RIGHT_OPERATION,GetCortexMetric(RESOLUTION_X),GetCortexMetric(RESOLUTION_Y),3),3 );
    } else
    if ( last_time_of_right_depth_map != VisCortx_GetVideoRegisterData(DEPTH_RIGHT_VIDEO,0) )
    {
         last_time_of_right_depth_map= VisCortx_GetVideoRegisterData(DEPTH_RIGHT_VIDEO,0);
         PassVideoRegisterToFeed ( 3 , VisCortx_ReadFromVideoRegister(DEPTH_RIGHT_VIDEO,GetCortexMetric(RESOLUTION_X),GetCortexMetric(RESOLUTION_Y),3),3 );
    }


*/

  return 1 ;
}


inline wxString _U(const char String[] = "")
{
  return wxString(String, wxConvUTF8);
}

