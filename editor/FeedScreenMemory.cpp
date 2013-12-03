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


void passVideoRegisterToFeed(unsigned int feednum,void * framedata,unsigned int width , unsigned int height , unsigned int bitsperpixel , unsigned int channels)
{
  if (framedata == 0 ) { fprintf(stderr,"Cannot passVideoRegisterToFeed for feed %u with null framedata\n",feednum); return; }
  if ( live_feeds[feednum].bmp == 0 ) {  return;  } else
  if ( live_feeds[feednum].bmp_allocated ) { delete live_feeds[feednum].bmp; live_feeds[feednum].bmp_allocated = false; live_feeds[feednum].bmp=0;}

  live_feeds[feednum].width=width;
  live_feeds[feednum].height=height;

 if ( framedata != 0)
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





inline wxString _U(const char String[] = "")
{
  return wxString(String, wxConvUTF8);
}

