#ifndef FEEDSCREENMEMORY_H_INCLUDED
#define FEEDSCREENMEMORY_H_INCLUDED

#include <wx/wx.h>

struct feed
{
   unsigned char feed_type , bitsPerPixel , channels;
   unsigned int width , height;
   unsigned short x1,y1,x2,y2;
   //wxMemoryDC *memDC;
   void *frame;
   wxImage img;
   wxBitmap *bmp;
   bool bmp_allocated;
};

extern feed live_feeds[4];
extern wxBitmap *default_feed;
extern wxBitmap *default_patch;
extern int has_init;
extern int VideoFeedsNotAccessible;


void initFeeds();
void closeFeeds();

#endif // FEEDSCREENMEMORY_H_INCLUDED
