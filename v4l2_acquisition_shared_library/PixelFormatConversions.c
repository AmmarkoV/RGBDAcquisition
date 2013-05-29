#include "PixelFormatConversions.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <linux/videodev2.h>

#include "V4L2Wrapper.h"




int VideoFormatNeedsDecoding(int videoformat,int bitdepth)
{
   switch (videoformat)
   {
      case V4L2_PIX_FMT_YUYV:
       return 1;
      break;
      case V4L2_PIX_FMT_RGB24:
       /* INPUT IS ALREADY RGB24 */
       return 0;
      break;
      default :
       printf("Video Format Needs Decoding but no handler implemnted, will propably output garbage on screen :S \n");
       return 0;
      break;
   };
  return 0;
}

int VideoFormatImplemented(int videoformat,int bitdepth)
{
   switch (videoformat)
   {
      case V4L2_PIX_FMT_YUYV:
       return 1;
      break;
      case V4L2_PIX_FMT_RGB24:
       /* INPUT IS ALREADY RGB24 */
       return 1;
      break;
      default :
       printf("Video Conversion Not Implemented, need to add handler for format , will propably output garbage on screen :S \n");
       return 0;
      break;
   };
  return 0;
}



void DebugSay(char * what)
{
 printf(" %s\n",what);
 return;
}

void PrintOutPixelFormat(int pix_format)
{
switch (pix_format)
     {
         case V4L2_PIX_FMT_YUYV :
          DebugSay((char *)"Setting pixel format to YUYV");
         break;
         case V4L2_PIX_FMT_VYUY :
          DebugSay((char *)"Setting pixel format to VYUY");
         break;
         case V4L2_PIX_FMT_YUV420 :
          DebugSay((char *)"Setting pixel format to YUV420");
         break;
         case V4L2_PIX_FMT_RGB24 :
          DebugSay((char *)"Setting pixel format to RGB24");
         break;
         case V4L2_PIX_FMT_BGR24 :
          DebugSay((char *)"Setting pixel format to BGR24");
         break;
         case V4L2_PIX_FMT_RGB32 :
          DebugSay((char *)"Setting pixel format to RGB32 ");
         break;
         case V4L2_PIX_FMT_YUV32 :
          DebugSay((char *)"Setting pixel format to YUV32 ");
         break;

         case V4L2_PIX_FMT_MJPEG :
          DebugSay((char *)"Setting pixel format to compressed MJPEG");
         break;
         case V4L2_PIX_FMT_JPEG :
          DebugSay((char *)"Setting pixel format to compressed JPEG");
         break;
         case V4L2_PIX_FMT_DV :
          DebugSay((char *)"Setting pixel format to compressed DV");
         break;
         case V4L2_PIX_FMT_MPEG :
          DebugSay((char *)"Setting pixel format to compressed MPEG ");
         break;

         default :
          DebugSay((char *)"Todo add pixel format to list :P");
         break;
     };
}

void PrintOutCaptureMode(int cap_mode)
{
switch (cap_mode)
     {
         case V4L2_BUF_TYPE_VIDEO_CAPTURE :
          DebugSay((char *)"Setting capture mode to Video Capture");
         break;
         case V4L2_BUF_TYPE_VIDEO_OUTPUT :
          DebugSay((char *)"Setting capture mode to Video Capture");
         break;
         case V4L2_BUF_TYPE_VIDEO_OVERLAY :
          DebugSay((char *)"Setting capture mode to Video Capture");
         break;
         case V4L2_BUF_TYPE_VBI_CAPTURE :
          DebugSay((char *)"Setting capture mode to Video Capture");
         break;
         case V4L2_BUF_TYPE_VBI_OUTPUT :
          DebugSay((char *)"Setting capture mode to Video Capture");
         break;
         case V4L2_BUF_TYPE_SLICED_VBI_CAPTURE :
          DebugSay((char *)"Setting capture mode to Video Capture");
         break;
         case V4L2_BUF_TYPE_SLICED_VBI_OUTPUT :
          DebugSay((char *)"Setting capture mode to Video Capture");
         break;
     };
}

void PrintOutFieldType(int field_type)
{
     switch (field_type)
     {
         case V4L2_FIELD_TOP :
          DebugSay((char *)"Setting pixel field to V4L2_FIELD_TOP");
         break;
         case V4L2_FIELD_INTERLACED :
          DebugSay((char *)"Setting pixel field to V4L2_FIELD_INTERLACED");
         break;
         case V4L2_FIELD_INTERLACED_TB :
          DebugSay((char *)"Setting pixel field to V4L2_FIELD_INTERLACED_TB");
         break;
         case V4L2_FIELD_INTERLACED_BT :
          DebugSay((char *)"Setting pixel field to V4L2_FIELD_INTERLACED_BT");
         break;
         case V4L2_FIELD_SEQ_TB :
          DebugSay((char *)"Setting pixel field to V4L2_FIELD_SEQ_TB");
         break;
         case V4L2_FIELD_SEQ_BT :
          DebugSay((char *)"Setting pixel field to V4L2_FIELD_SEQ_BT");
         break;

         default :
          DebugSay((char *)"Todo: Add video field type to list");
         break;
     };
}



/*
  -----------------------------------------
    This is the central converter that should handle frame ENC_frame with videoformat / bitdepth
    to an RGB24 frame ( allocated @ RGB_frame )
    Only YUYV is implemented , I have 4 webcams and they are all RGB24 or YUYV
    You should add the handler
  -----------------------------------------
*/
int Convert2RGB24(unsigned char * ENC_frame , unsigned char * RGB_frame,unsigned int width,unsigned int height,int inp_videoformat,int inp_bitdepth)
{
  if ( ( ENC_frame == 0 ) || ( RGB_frame == 0 ) ) { return 0; }

  /* This function will pick an appropriate decoder to change signal from ENC to RGB24*/
  switch (inp_videoformat)
   {
      case V4L2_PIX_FMT_YUYV:
       if ( inp_bitdepth==16 ) { YUYV_2_RGB(ENC_frame,RGB_frame,width,height); }
       return 1;
      break;
      /*
         NEEDS MORE HANDLERS HERE
      */
      default :
       printf("Unhandled format :S needs addition here\n");
      break;
   };
 return 0;
}


int YUYVY_ImplementationCheck_OK()
{
  if ( sizeof(unsigned int)!=4 ) { fprintf(stderr,"YUYVY_Implementation ERROR unsigned int not 4 bytes long!!\n"); }
  return 1;
}

/*
  -----------------------------------------
    YUYV -> RGB24 Conversion following..!
    Next two functions found , at http://www.quickcamteam.net/ written by Logitech
  -----------------------------------------
*/
static inline int convert_yuv_to_rgb_pixel(int y, int u, int v)
{
	unsigned int pixel32 = 0;
	unsigned char *pixel = (unsigned char *)&pixel32;
	int r, g, b;

	r = y + (1.370705 * (v-128));
	g = y - (0.698001 * (v-128)) - (0.337633 * (u-128));
	b = y + (1.732446 * (u-128));

	if(r > 255) r = 255;
	if(g > 255) g = 255;
	if(b > 255) b = 255;
	if(r < 0) r = 0;
	if(g < 0) g = 0;
	if(b < 0) b = 0;

    /*
    This seems to be a little faster :P
    1000-2533 fps for every 30 frames :P
    */
	pixel[0] = r * 0.859375;
	pixel[1] = g * 0.859375;
	pixel[2] = b * 0.859375;

    /*  833-1033  fps for every 30 frames :P  */
/*  pixel[0] = r * 220 / 256;
	pixel[1] = g * 220 / 256;
	pixel[2] = b * 220 / 256;*/

	return pixel32;
}


int YUYV_2_RGB(unsigned char *yuv, unsigned char *rgb, unsigned int width, unsigned int height)
{
	unsigned int in, out = 0;
	unsigned int pixel_16;
	unsigned char pixel_24[3];
	unsigned int pixel32;
	int y0, u, y1, v;

	for(in = 0; in < width * height * 2; in += 4) {
		pixel_16 =
			yuv[in + 3] << 24 |
			yuv[in + 2] << 16 |
			yuv[in + 1] <<  8 |
			yuv[in + 0];

		y0 = (pixel_16 & 0x000000ff);
		u  = (pixel_16 & 0x0000ff00) >>  8;
		y1 = (pixel_16 & 0x00ff0000) >> 16;
		v  = (pixel_16 & 0xff000000) >> 24;

		pixel32 = convert_yuv_to_rgb_pixel(y0, u, v);
		pixel_24[0] = (pixel32 & 0x000000ff);
		pixel_24[1] = (pixel32 & 0x0000ff00) >> 8;
		pixel_24[2] = (pixel32 & 0x00ff0000) >> 16;

		rgb[out++] = pixel_24[0];
		rgb[out++] = pixel_24[1];
		rgb[out++] = pixel_24[2];

		pixel32 = convert_yuv_to_rgb_pixel(y1, u, v);
		pixel_24[0] = (pixel32 & 0x000000ff);
		pixel_24[1] = (pixel32 & 0x0000ff00) >> 8;
		pixel_24[2] = (pixel32 & 0x00ff0000) >> 16;

		rgb[out++] = pixel_24[0];
		rgb[out++] = pixel_24[1];
		rgb[out++] = pixel_24[2];
	}
	return 0;
}




/*
 * Turn a YUV4:2:0 block into an RGB block
 *
 * Video4Linux seems to use the blue, green, red channel
 * order convention-- rgb[0] is blue, rgb[1] is green, rgb[2] is red.
 *
 * Color space conversion coefficients taken from the excellent
 * http://www.inforamp.net/~poynton/ColorFAQ.html
 * In his terminology, this is a CCIR 601.1 YCbCr -> RGB.
 * Y values are given for all 4 pixels, but the U (Pb)
 * and V (Pr) are assumed constant over the 2x2 block.
 *
 * To avoid floating point arithmetic, the color conversion
 * coefficients are scaled into 16.16 fixed-point integers.
 * They were determined as follows:
 *
 *  double brightness = 1.0;  (0->black; 1->full scale)
 *  double saturation = 1.0;  (0->greyscale; 1->full color)
 *  double fixScale = brightness * 256 * 256;
 *  int rvScale = (int)(1.402 * saturation * fixScale);
 *  int guScale = (int)(-0.344136 * saturation * fixScale);
 *  int gvScale = (int)(-0.714136 * saturation * fixScale);
 *  int buScale = (int)(1.772 * saturation * fixScale);
 *  int yScale = (int)(fixScale);
 */

/* LIMIT: convert a 16.16 fixed-point value to a byte, with clipping. */
#define LIMIT(x) ((x)>0xffffff?0xff: ((x)<=0xffff?0:((x)>>16)))

static inline void
move_420_block(int yTL, int yTR, int yBL, int yBR, int u, int v,
           int rowPixels, unsigned char * rgb)
{
    const int rvScale = 91881;
    const int guScale = -22553;
    const int gvScale = -46801;
    const int buScale = 116129;
    const int yScale  = 65536;
    int r, g, b;

    g = guScale * u + gvScale * v;
/*  if (force_rgb) {
      r = buScale * u;
      b = rvScale * v;
  } else {*/
        r = rvScale * v;
        b = buScale * u;
/*  }*/

    yTL *= yScale; yTR *= yScale;
    yBL *= yScale; yBR *= yScale;

    /* Write out top two pixels */
    rgb[0] = LIMIT(r+yTL); rgb[1] = LIMIT(g+yTL);
    rgb[2] = LIMIT(b+yTL);

    rgb[3] = LIMIT(r+yTR); rgb[4] = LIMIT(g+yTR);
    rgb[5] = LIMIT(b+yTR);

    /* Skip down to next line to write out bottom two pixels */
    rgb += 3 * rowPixels;
    rgb[0] = LIMIT(r+yBL); rgb[1] = LIMIT(g+yBL);
    rgb[2] = LIMIT(b+yBL);

    rgb[3] = LIMIT(r+yBR); rgb[4] = LIMIT(g+yBR);
    rgb[5] = LIMIT(b+yBR);
}

static inline void
move_411_block(int yTL, int yTR, int yBL, int yBR, int u, int v,
           int rowPixels, unsigned char * rgb)
{
    const int rvScale = 91881;
    const int guScale = -22553;
    const int gvScale = -46801;
    const int buScale = 116129;
    const int yScale  = 65536;
    int r, g, b;

    g = guScale * u + gvScale * v;
/*  if (force_rgb) {
*      r = buScale * u;
*      b = rvScale * v;
*  } else {*/
        r = rvScale * v;
        b = buScale * u;
/*  } */

    yTL *= yScale; yTR *= yScale;
    yBL *= yScale; yBR *= yScale;

    /* Write out top two first pixels */
    rgb[0] = LIMIT(r+yTL); rgb[1] = LIMIT(g+yTL);
    rgb[2] = LIMIT(b+yTL);

    rgb[3] = LIMIT(r+yTR); rgb[4] = LIMIT(g+yTR);
    rgb[5] = LIMIT(b+yTR);

    /* Write out top two last pixels */
    rgb += 6;
    rgb[0] = LIMIT(r+yBL); rgb[1] = LIMIT(g+yBL);
    rgb[2] = LIMIT(b+yBL);

    rgb[3] = LIMIT(r+yBR); rgb[4] = LIMIT(g+yBR);
    rgb[5] = LIMIT(b+yBR);
}

/* Consider a YUV420P image of 8x2 pixels.

 A plane of Y values    A B C D E F G H
                        I J K L M N O P

 A plane of U values    1   2   3   4
 A plane of V values    1   2   3   4 ....

 The U1/V1 samples correspond to the ABIJ pixels.
     U2/V2 samples correspond to the CDKL pixels.

 Converts from planar YUV420P to RGB24. */
unsigned char *yuv420p_to_rgb24(int width, int height,
				unsigned char *pIn0, unsigned char *pOut0)
{


    const int numpix = width * height;
    const int bytes = 24 >> 3;
    if (pOut0==0)
    {
      pOut0=(unsigned char *)malloc(numpix*3);
    }
    int i, j, y00, y01, y10, y11, u, v;
    unsigned char *pY = pIn0;
    unsigned char *pU = pY + numpix;
    unsigned char *pV = pU + numpix / 4;
    unsigned char *pOut = pOut0;

    for (j = 0; j <= height - 2; j += 2) {
        for (i = 0; i <= width - 2; i += 2) {
            y00 = *pY;
            y01 = *(pY + 1);
            y10 = *(pY + width);
            y11 = *(pY + width + 1);
            u = (*pU++) - 128;
            v = (*pV++) - 128;

            move_420_block(y00, y01, y10, y11, u, v,
                       width, pOut);

            pY += 2;
            pOut += 2 * bytes;

        }
        pY += width;
        pOut += width * bytes;
    }
    return pOut0;
}

/* Consider a YUV420 image of 6x2 pixels.

 A B C D U1 U2
 I J K L V1 V2

 The U1/V1 samples correspond to the ABIJ pixels.
     U2/V2 samples correspond to the CDKL pixels.

 Converts from interlaced YUV420 to RGB24. */
/* [FD] untested... */
unsigned char *yuv420_to_rgb24(int width, int height,
			       unsigned char *pIn0, unsigned char *pOut0)
{
    const int numpix = width * height;
    const int bytes = 24 >> 3;
    if (pOut0==0)
    {
      pOut0=(unsigned char *)malloc(numpix*3);
    }
    int i, j, y00, y01, y10, y11, u, v;
    unsigned char *pY = pIn0;
    unsigned char *pU = pY + 4;
    unsigned char *pV = pU + width;
    unsigned char *pOut = pOut0;

    for (j = 0; j <= height - 2; j += 2) {
        for (i = 0; i <= width - 4; i += 4) {
            y00 = *pY;
            y01 = *(pY + 1);
            y10 = *(pY + width);
            y11 = *(pY + width + 1);
            u = (*pU++) - 128;
            v = (*pV++) - 128;

            move_420_block(y00, y01, y10, y11, u, v,
                       width, pOut);

            pY += 2;
            pOut += 2 * bytes;

            y00 = *pY;
            y01 = *(pY + 1);
            y10 = *(pY + width);
            y11 = *(pY + width + 1);
            u = (*pU++) - 128;
            v = (*pV++) - 128;

            move_420_block(y00, y01, y10, y11, u, v,
                       width, pOut);

            pY += 4; /* skip UV */
            pOut += 2 * bytes;

        }
        pY += width;
        pOut += width * bytes;
    }
    return pOut0;
}

/* Consider a YUV411P image of 8x2 pixels.
*
* A plane of Y values as before.
*
* A plane of U values    1       2
*                        3       4
*
* A plane of V values    1       2
*                        3       4
*
* The U1/V1 samples correspond to the ABCD pixels.
*     U2/V2 samples correspond to the EFGH pixels.

 Converts from planar YUV411P to RGB24. */
/* [FD] untested... */
unsigned char *yuv411p_to_rgb24(int width, int height,
				unsigned char *pIn0, unsigned char *pOut0)
{
    const int numpix = width * height;
    const int bytes = 24 >> 3;
    if (pOut0==0)
    {
      pOut0=(unsigned char *)malloc(numpix*3);
    }
    int i, j, y00, y01, y10, y11, u, v;
    unsigned char *pY = pIn0;
    unsigned char *pU = pY + numpix;
    unsigned char *pV = pU + numpix / 4;
    unsigned char *pOut = pOut0;

    for (j = 0; j <= height; j++) {
        for (i = 0; i <= width - 4; i += 4) {
            y00 = *pY;
            y01 = *(pY + 1);
            y10 = *(pY + 2);
            y11 = *(pY + 3);
            u = (*pU++) - 128;
            v = (*pV++) - 128;

            move_411_block(y00, y01, y10, y11, u, v,
                       width, pOut);

            pY += 4;
            pOut += 4 * bytes;

        }
    }
    return pOut0;
}
