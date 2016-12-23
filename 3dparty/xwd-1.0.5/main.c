/*

Copyright 1987, 1998  The Open Group

Permission to use, copy, modify, distribute, and sell this software and its
documentation for any purpose is hereby granted without fee, provided that
the above copyright notice appear in all copies and that both that
copyright notice and this permission notice appear in supporting
documentation.

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
OPEN GROUP BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Except as contained in this notice, the name of The Open Group shall not be
used in advertising or otherwise to promote the sale, use or other dealings
in this Software without prior written authorization from The Open Group.

*/

/*
 * xwd.c MIT Project Athena, X Window system window raster image dumper.
 *
 * This program will dump a raster image of the contents of a window into a
 * file for output on graphics printers or for other uses.
 *
 *  Author:	Tony Della Fera, DEC
 *		17-Jun-85
 *
 *  Modification history:
 *
 *  11/14/86 Bill Wyatt, Smithsonian Astrophysical Observatory
 *    - Removed Z format option, changing it to an XY option. Monochrome
 *      windows will always dump in XY format. Color windows will dump
 *      in Z format by default, but can be dumped in XY format with the
 *      -xy option.
 *
 *  11/18/86 Bill Wyatt
 *    - VERSION 6 is same as version 5 for monchrome. For colors, the
 *      appropriate number of Color structs are dumped after the header,
 *      which has the number of colors (=0 for monochrome) in place of the
 *      V5 padding at the end. Up to 16-bit displays are supported. I
 *      don't yet know how 24- to 32-bit displays will be handled under
 *      the Version 11 protocol.
 *
 *  6/15/87 David Krikorian, MIT Project Athena
 *    - VERSION 7 runs under the X Version 11 servers, while the previous
 *      versions of xwd were are for X Version 10.  This version is based
 *      on xwd version 6, and should eventually have the same color
 *      abilities. (Xwd V7 has yet to be tested on a color machine, so
 *      all color-related code is commented out until color support
 *      becomes practical.)
 */

/*%
 *%    This is the format for commenting out color-related code until
 *%  color can be supported.
%*/

#include <stdio.h>
#include <errno.h>
#include <X11/Xos.h>
#include <stdlib.h>

#include <X11/Xlib.h>
#include <X11/Xutil.h>

typedef unsigned long Pixel;
#include "X11/XWDFile.h"

#define FEEP_VOLUME 0

/* Include routines to do parsing */
#include "dsimple.h"
#include "list.h"
#include "wsutils.h"
#include "multiVis.h"

#ifdef XKB
#include <X11/extensions/XKBbells.h>
#endif

/* Setable Options */

static int format = ZPixmap;
static Bool nobdrs = False;
static Bool on_root = False;
static Bool standard_out = True;
static Bool debug = False;
static Bool silent = False;
static Bool use_installed = False;
static long add_pixel_value = 0;


extern int main(int, char **);
extern void Window_Dump(Window window, FILE *out , unsigned char * data , unsigned int * dataWidth , unsigned int * dataHeight);
extern int Image_Size(XImage *);
extern int Get_XColors(XWindowAttributes *, XColor **);
extern void _swapshort(register char *, register unsigned);
extern void _swaplong(register char *, register unsigned);
static long parse_long(char *);
static int Get24bitDirectColors(XColor **);
static int ReadColors(Visual *, Colormap, XColor **);

     int i;
    Window target_win;
    FILE *out_file = 0;
    Bool frame_only = False;

int initXwdLib(int argc, char  **argv)
{
    //Setup_Display_And_Screen(argc,argv);
    Setup_Null_Display_And_Screen();
}


int closeXwdLib()
{
    XCloseDisplay(dpy);
}

int getScreen(unsigned char * frame , unsigned int * frameWidth , unsigned int * frameHeight)
{
    target_win=getRootWindow();

    /* Let the user select the target window. */
    if (target_win == None) target_win = Select_Window(dpy, !frame_only);

    /* Dump it! */
    Window_Dump(target_win, stdout , frame , frameWidth , frameHeight);

}

static int
Get24bitDirectColors(XColor **colors)
{
    int i , ncolors = 256 ;
    XColor *tcol ;

    *colors = tcol = (XColor *)malloc(sizeof(XColor) * ncolors) ;

    for(i=0 ; i < ncolors ; i++)
    {
	tcol[i].pixel = i << 16 | i << 8 | i ;
	tcol[i].red = tcol[i].green = tcol[i].blue = i << 8   | i ;
    }

    return ncolors ;
}


/*
 * Window_Dump: dump a window to a file which must already be open for
 *              writting.
 */




void
Window_Dump(Window window, FILE *out , unsigned char * data , unsigned int * dataWidth , unsigned int * dataHeight)
{
    unsigned long swaptest = 1;
    XColor *colors;
    unsigned buffer_size;
    int win_name_size;
    int header_size;
    int ncolors, i;
    char *win_name;
    Bool got_win_name;
    XWindowAttributes win_info;
    XImage *image;
    int absx, absy, x, y;
    unsigned width, height;
    int dwidth, dheight;
    Window dummywin;
    XWDFileHeader header;
    XWDColor xwdcolor;

    int                 transparentOverlays , multiVis;
    int                 numVisuals;
    XVisualInfo         *pVisuals;
    int                 numOverlayVisuals;
    OverlayInfo         *pOverlayVisuals;
    int                 numImageVisuals;
    XVisualInfo         **pImageVisuals;
    list_ptr            vis_regions;    /* list of regions to read from */
    list_ptr            vis_image_regions ;
    Visual		vis_h,*vis ;
    int			allImage = 0 ;

    /*
     * Inform the user not to alter the screen.
     */
    if (!silent) {
#ifdef XKB
	XkbStdBell(dpy,None,50,XkbBI_Wait);
#else
	XBell(dpy,FEEP_VOLUME);
#endif
	XFlush(dpy);
    }

    /*
     * Get the parameters of the window being dumped.
     */
    if (debug) outl("xwd: Getting target window information.\n");
    if(!XGetWindowAttributes(dpy, window, &win_info))
      Fatal_Error("Can't get target window attributes.");

    /* handle any frame window */
    if (!XTranslateCoordinates (dpy, window, RootWindow (dpy, screen), 0, 0,
				&absx, &absy, &dummywin)) {
	fprintf (stderr,
		 "%s:  unable to translate window coordinates (%d,%d)\n",
		 program_name, absx, absy);
	exit (1);
    }
    win_info.x = absx;
    win_info.y = absy;
    width = win_info.width;
    height = win_info.height;

    if (!nobdrs)  {
                    absx -= win_info.border_width;
	                absy -= win_info.border_width;
	                width += (2 * win_info.border_width);
	                height += (2 * win_info.border_width);
                  }
    dwidth = DisplayWidth (dpy, screen);
    dheight = DisplayHeight (dpy, screen);


    /* clip to window */
    if (absx < 0) width += absx, absx = 0;
    if (absy < 0) height += absy, absy = 0;
    if (absx + width > dwidth) width = dwidth - absx;
    if (absy + height > dheight) height = dheight - absy;

    XFetchName(dpy, window, &win_name);
    if (!win_name || !win_name[0]) { win_name = "xwdump"; got_win_name = False; } else
                                   { got_win_name = True; }

    /* sizeof(char) is included for the null string terminator. */
    win_name_size = strlen(win_name) + sizeof(char);

    /*
     * Snarf the pixmap with XGetImage.
     */

    x = absx - win_info.x;
    y = absy - win_info.y;

    image = XGetImage (dpy, RootWindow(dpy, screen), absx, absy, width, height, AllPlanes, format);

    /*
    multiVis = GetMultiVisualRegions(dpy,RootWindow(dpy, screen),
               absx, absy,
	       width, height,&transparentOverlays,&numVisuals, &pVisuals,
               &numOverlayVisuals,&pOverlayVisuals,&numImageVisuals,
               &pImageVisuals,&vis_regions,&vis_image_regions,&allImage) ;
    if (on_root || multiVis)
    {
	if(!multiVis)
	    image = XGetImage (dpy, RootWindow(dpy, screen), absx, absy,
                    width, height, AllPlanes, format);
	else
	    image = ReadAreaToImage(dpy, RootWindow(dpy, screen), absx, absy,
                width, height,
    		numVisuals,pVisuals,numOverlayVisuals,pOverlayVisuals,
                numImageVisuals, pImageVisuals,vis_regions,
                vis_image_regions,format,allImage);
    }
    else
	image = XGetImage (dpy, window, x, y, width, height, AllPlanes, format);
    if (!image) {
	fprintf (stderr, "%s:  unable to get image at %dx%d+%d+%d\n",
		 program_name, width, height, x, y);
	exit (1);
    }
*/


    unsigned int r,g,b;
    unsigned int pixelvalue;

    /*
     * Write out the buffer.
     */
    if (debug) outl("xwd: Dumping pixmap.  bufsize=%d\n",buffer_size);

    *dataWidth =image->width;
    *dataHeight=image->height;
   // memcpy(data,image->data ,image->width * image->height * 3 );

  if ( image->bits_per_pixel/8!=sizeof(int) )
  {
      fprintf(stderr,"BitsPerPixel != sizeof(int)    -  %u != %u \n",image->bits_per_pixel,sizeof(int));
  }

  unsigned char * targetPtr = data;
  unsigned int * line_ptr;

  unsigned long pixel;
  for ( y=0; y<image->height-1; y++)
     {
      for (x=0; x<image->width; x++)
       {
           pixel = XGetPixel (image,x,y);

           *targetPtr = (pixel >> 16) & 0xff; ++targetPtr;
           *targetPtr = (pixel >>  8) & 0xff; ++targetPtr;
           *targetPtr = (pixel >>  0) & 0xff; ++targetPtr;
        }
     }

    /*
     * Free image
     */
    XDestroyImage(image);
}

/*
 * Report the syntax for calling xwd.
 */
void
usage(void)
{
    fprintf (stderr,
"usage: %s [-display host:dpy] [-debug] [-help] %s [-nobdrs] [-out <file>]",
	   program_name, "[{-root|-id <id>|-name <name>}]");
    fprintf (stderr, " [-xy] [-add value] [-frame]\n");
    exit(1);
}


/*
 * Determine the pixmap size.
 */

int Image_Size(XImage *image)
{
    if (image->format != ZPixmap)
      return(image->bytes_per_line * image->height * image->depth);

    return(image->bytes_per_line * image->height);
}

#define lowbit(x) ((x) & (~(x) + 1))

static int
ReadColors(Visual *vis, Colormap cmap, XColor **colors)
{
    int i,ncolors ;

    ncolors = vis->map_entries;

    if (!(*colors = (XColor *) malloc (sizeof(XColor) * ncolors)))
      Fatal_Error("Out of memory!");

    if (vis->class == DirectColor ||
	vis->class == TrueColor) {
	Pixel red, green, blue, red1, green1, blue1;

	red = green = blue = 0;
	red1 = lowbit(vis->red_mask);
	green1 = lowbit(vis->green_mask);
	blue1 = lowbit(vis->blue_mask);
	for (i=0; i<ncolors; i++) {
	  (*colors)[i].pixel = red|green|blue;
	  (*colors)[i].pad = 0;
	  red += red1;
	  if (red > vis->red_mask)
	    red = 0;
	  green += green1;
	  if (green > vis->green_mask)
	    green = 0;
	  blue += blue1;
	  if (blue > vis->blue_mask)
	    blue = 0;
	}
    } else {
	for (i=0; i<ncolors; i++) {
	  (*colors)[i].pixel = i;
	  (*colors)[i].pad = 0;
	}
    }

    XQueryColors(dpy, cmap, *colors, ncolors);

    return(ncolors);
}

/*
 * Get the XColors of all pixels in image - returns # of colors
 */
int Get_XColors(XWindowAttributes *win_info, XColor **colors)
{
    int i, ncolors;
    Colormap cmap = win_info->colormap;

    if (use_installed)
	/* assume the visual will be OK ... */
	cmap = XListInstalledColormaps(dpy, win_info->root, &i)[0];
    if (!cmap)
	return(0);
    ncolors = ReadColors(win_info->visual,cmap,colors) ;
    return ncolors ;
}

void
_swapshort (register char *bp, register unsigned n)
{
    register char c;
    register char *ep = bp + n;

    while (bp < ep) {
	c = *bp;
	*bp = *(bp + 1);
	bp++;
	*bp++ = c;
    }
}

void
_swaplong (register char *bp, register unsigned n)
{
    register char c;
    register char *ep = bp + n;

    while (bp < ep) {
        c = bp[3];
        bp[3] = bp[0];
        bp[0] = c;
        c = bp[2];
        bp[2] = bp[1];
        bp[1] = c;
        bp += 4;
    }
}
