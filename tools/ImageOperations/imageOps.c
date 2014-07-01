#include "imageOps.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ABSDIFF(num1,num2) ( (num1-num2) >=0 ? (num1-num2) : (num2 - num1) )

#define MEMPLACE1(x,y,width) ( y * ( width  ) + x )
#define MEMPLACE3(x,y,width) ( y * ( width * 3 ) + x*3 )

#define MIN2(A,B)       ((A)<(B)?(A):(B))
#define MIN3(A,B,C)     (MIN2(MIN2((A),(B)),(C)))

#define MAX2(A,B)       ((A)>(B)?(A):(B))
#define MAX3(A,B,C)     (MAX2(MAX2((A),(B)),(C)))


unsigned short getDepthValueAtXY(unsigned short * depthFrame ,unsigned int width , unsigned int height ,unsigned int x2d, unsigned int y2d )
{
    if (depthFrame == 0 ) {  return 0; }
    if ( (x2d>=width) || (y2d>=height) )    {   return 0; }


    unsigned short * depthValue = depthFrame + (y2d * width + x2d );
    unsigned short result = * depthValue;

    return result;
}

void setDepthValueAtXY(unsigned short * depthFrame ,unsigned int width , unsigned int height ,unsigned int x2d, unsigned int y2d , unsigned int value )
{
    if (depthFrame == 0 ) {  return 0; }
    if ( (x2d>=width) || (y2d>=height) )    {   return 0; }

    unsigned short * depthValue = depthFrame + (y2d * width + x2d );
    *depthValue  = value;
}



int trainDepthClassifier(struct depthClassifier * dc ,
                         unsigned short * target,  unsigned int tX,  unsigned int tY  , unsigned int targetWidth , unsigned int targetHeight ,
                         unsigned int tileWidth , unsigned int tileHeight)
{
  unsigned int depthValue=0;
  unsigned int i=0;

  unsigned int useX,useY;
  float rateX=1.0,rateY=1.0;
  dc->totalSamples=0;

  for (i=0; i<dc->currentPointList; i++)
  {
     rateX = (float ) dc->pointList[i].x / dc->patchWidth;
     useX = tileWidth * rateX;

     rateY = (float ) dc->pointList[i].y / dc->patchHeight;
     useY = tileHeight * rateY;

     depthValue=getDepthValueAtXY(target,targetWidth,targetHeight ,  tX + useX , tY + useY );

     if (depthValue!=0)
     {
       if (dc->totalSamples==0) { dc->depthBase=depthValue; ++dc->totalSamples; } else
                                             {
                                               if (dc->depthBase > depthValue) { dc->depthBase=depthValue; }
                                               ++dc->totalSamples;
                                             }

      dc->pointList[i].depth=depthValue-dc->depthBase;

      if ( dc->pointList[i].samples==0)
       {
       dc->pointList[i].minAccepted = dc->pointList[i].depth;
       dc->pointList[i].maxAccepted = dc->pointList[i].depth;
        ++dc->pointList[i].samples;
       } else
       {
        if (dc->pointList[i].minAccepted>dc->pointList[i].depth) { dc->pointList[i].minAccepted=dc->pointList[i].depth; }
        if (dc->pointList[i].maxAccepted<dc->pointList[i].depth) { dc->pointList[i].maxAccepted=dc->pointList[i].depth; }
        ++dc->pointList[i].samples;
       }
     }
  }


  for (i=0; i<dc->currentPointList; i++)
  {
      if ( dc->pointList[i].samples!=0)
       {
           dc->pointList[i].depth=dc->pointList[i].depth-dc->depthBase;
       }
  }



 return 1;
}





unsigned int compareDepthClassifier(struct depthClassifier * dc ,
                                    unsigned short * target,  unsigned int tX,  unsigned int tY  , unsigned int targetWidth , unsigned int targetHeight ,
                                    unsigned int tileWidth , unsigned int tileHeight)
{
  unsigned int totalDepthMismatch = 0;
  unsigned int i=0;

  dc->totalSamples=0;

  unsigned int useX,useY;
  float rateX=1.0,rateY=1.0;

  for (i=0; i<dc->currentPointList; i++)
  {
     rateX = (float ) dc->pointList[i].x / dc->patchWidth;
     useX = tileWidth * rateX;

     rateY = (float ) dc->pointList[i].y / dc->patchHeight;
     useY = tileHeight * rateY;



     dc->pointList[i].depth = getDepthValueAtXY(target,targetWidth,targetHeight ,  tX + useX , tY + useY );

     setDepthValueAtXY(target,targetWidth,targetHeight,tX + useX,tY + useY,65536);

     if (dc->totalSamples==0) { dc->depthBase = dc->pointList[i].depth; ++dc->totalSamples; } else
                              {
                                if (dc->depthBase >  dc->pointList[i].depth) { dc->depthBase= dc->pointList[i].depth; }
                                ++dc->totalSamples;
                              }
  }


  for (i=0; i<dc->currentPointList; i++)
  {
     if ( (dc->pointList[i].depth!=0) && ( dc->pointList[i].samples>1) )
     {
       dc->pointList[i].depth=dc->pointList[i].depth-dc->depthBase;

       if (dc->pointList[i].maxAccepted<dc->pointList[i].depth) { totalDepthMismatch+=dc->pointList[i].depth - dc->pointList[i].maxAccepted; } else
       if (dc->pointList[i].minAccepted>dc->pointList[i].depth) { totalDepthMismatch+=dc->pointList[i].minAccepted - dc->pointList[i].depth; }
     }
  }
 fprintf(stderr,"Depth Classifier returned %u\n",totalDepthMismatch);
 return totalDepthMismatch;
}



int printDepthClassifier(char * filename , struct depthClassifier * dc )
{
  FILE *fpr = 0;
  fpr=fopen(filename,"w");
  if (fpr!=0)
  {
   fprintf(fpr,"void initDepthClassifier(struct depthClassifier * dc)\n");
   fprintf(fpr,"{ \n");
    unsigned int i=0;
    fprintf(fpr,"dc->currentPointList=%u;\n",dc->currentPointList);
    fprintf(fpr,"dc->depthBase=%u;\n",dc->depthBase);
    fprintf(fpr,"dc->totalSamples=%u;\n",dc->totalSamples);
    fprintf(fpr,"dc->patchWidth=%u;\n",dc->patchWidth);
    fprintf(fpr,"dc->patchHeight=%u;\n",dc->patchHeight);
    for (i=0; i<dc->currentPointList; i++)
     {
       fprintf(fpr,"\ndc->pointList[%u].x=%u;\n",i,dc->pointList[i].x);
       fprintf(fpr,"dc->pointList[%u].y=%u;\n",i,dc->pointList[i].y);
       fprintf(fpr,"dc->pointList[%u].minAccepted=%u;\n",i,dc->pointList[i].minAccepted);
       fprintf(fpr,"dc->pointList[%u].maxAccepted=%u;\n",i,dc->pointList[i].maxAccepted);
       fprintf(fpr,"dc->pointList[%u].samples=%u;\n",i,dc->pointList[i].samples);
     }
    fprintf(fpr,"\n\n");
   fprintf(fpr,"}\n");

   fclose(fpr);
  }
 return 1;
}











//RGB 2 HSV / HSV 2 RGB from http://www.cs.rit.edu/~ncs/color/t_convert.html
// r,g,b values are from 0 to 1
// h = [0,360], s = [0,1], v = [0,1]
//		if s == 0, then h = -1 (undefined)

void RGBFtoHSV( float r, float g, float b, float *h, float *s, float *v )
{
	float min, max, delta;

	min = MIN3( r, g, b );
	max = MAX3( r, g, b );
	*v = max;				// v

	delta = max - min;

	if( max != 0 )
		*s = delta / max;		// s
	else {
		// r = g = b = 0		// s = 0, v is undefined
		*s = 0;
		*h = -1;
		return;
	}

	if( r == max )
		*h = ( g - b ) / delta;		// between yellow & magenta
	else if( g == max )
		*h = 2 + ( b - r ) / delta;	// between cyan & yellow
	else
		*h = 4 + ( r - g ) / delta;	// between magenta & cyan

	*h *= 60;				// degrees
	if( *h < 0 )
		*h += 360;

}


void RGBtoHSV( unsigned char r, unsigned char g, unsigned char b,
               float *h, float *s, float *v )
{
  float rInt = (float) r/255;
  float gInt = (float) g/255;
  float bInt = (float) b/255;
  RGBFtoHSV(rInt,gInt,bInt,h,s,v);
}


void HSVtoRGB( float *r, float *g, float *b, float h, float s, float v )
{
	int i;
	float f, p, q, t;

	if( s == 0 ) {
		// achromatic (grey)
		*r = *g = *b = v;
		return;
	}

	h /= 60;			// sector 0 to 5
	i = floor( h );
	f = h - i;			// factorial part of h
	p = v * ( 1 - s );
	q = v * ( 1 - s * f );
	t = v * ( 1 - s * ( 1 - f ) );

	switch( i ) {
		case 0:
			*r = v;
			*g = t;
			*b = p;
			break;
		case 1:
			*r = q;
			*g = v;
			*b = p;
			break;
		case 2:
			*r = p;
			*g = v;
			*b = t;
			break;
		case 3:
			*r = p;
			*g = q;
			*b = v;
			break;
		case 4:
			*r = t;
			*g = p;
			*b = v;
			break;
		default:		// case 5:
			*r = v;
			*g = p;
			*b = q;
			break;
	}

}
unsigned int simplePowInline(unsigned int base,unsigned int exp)
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



int saveRawImageToFile(char * filename,char *comments ,unsigned char * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel)
{
    //fprintf(stderr,"saveRawImageToFile(%s) called\n",filename);

    if ( (width==0) || (height==0) || (channels==0) || (bitsperpixel==0) ) { fprintf(stderr,"saveRawImageToFile(%s) called with zero dimensions\n",filename); return 0;}
    if(pixels==0) { fprintf(stderr,"saveRawImageToFile(%s) called for an unallocated (empty) frame , will not write any file output\n",filename); return 0; }
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

        if (comments!=0)
        {
          fprintf(fd, "#%s\n",comments);
        } else
        {
          fprintf(fd, "#generated by https://github.com/AmmarkoV/RGBDAcquisition/blob/master/tools/ImageOperations/imageOps.c\n");
        }



        fprintf(fd, "%d %d\n%u\n", width, height , simplePowInline(2 ,bitsperpixel)-1);

        float tmp_n = (float) bitsperpixel/ 8;
        tmp_n = tmp_n * width * height * channels ;
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






int shiftImageRGB(unsigned char * target, unsigned char * source ,  unsigned char transR, unsigned char transG, unsigned char transB , signed int tX,  signed int tY  ,  unsigned int width , unsigned int height)
{
  if ( (target==0)||(source==0) ) { return 0; }
  if ( (width==0)&&(height==0) ) { return 0; }


  unsigned int sourceWidth=width,sourceHeight=height,targetWidth=width,targetHeight=height;
  unsigned int sourceX,sourceY , targetX,targetY;

  if (tX < 0 ) {   sourceX=abs(tX);    targetX=0;   } else
               {   sourceX=0;          targetX=abs(tX); }

  if (tY < 0 ) { sourceY=abs(tY); targetY=0;  } else
               { sourceY=0;       targetY=abs(tY); }

  width=width - abs(tX);
  height=height - abs(tY);

  if (width>sourceWidth) { width=sourceWidth; fprintf(stderr,"Error setting width (?) why did this happen ? :P \n"); }
  if (height>sourceHeight) { height=sourceHeight; fprintf(stderr,"Error setting height (?) why did this happen ? :P \n"); }

  fprintf(stderr,"Doing shift (%d,%d) by bit blitting %u,%u to %u,%u ( size %u,%u) \n",tX,tY,sourceX,sourceY,targetX,targetY,width,height);


  //----------------------------------------------------------------
  unsigned char * maybeCopiedSource = source;
  //In case tx or ty is positive it is impossible to bit blt using the same buffer since lines will be overriden so we use a seperate buffer

  if ( (tX>0) || (tY>0) )
  {
    unsigned int copySize = (width+1)*(height+1)*3*sizeof(unsigned char);
    maybeCopiedSource = (unsigned char * ) malloc(copySize);
    if (maybeCopiedSource==0) { maybeCopiedSource = source; } else
                              { memcpy(maybeCopiedSource,source,copySize); }
  }
  //----------------------------------------------------------------

  bitbltRGB( target ,targetX,targetY, targetWidth,targetHeight,
             maybeCopiedSource ,sourceX,sourceY, sourceWidth,sourceHeight,
             width,height);

  //----------------------------------------------------------------
    if ( (tX>0) || (tY>0) )
   {
     if (maybeCopiedSource!=source) { free(maybeCopiedSource); maybeCopiedSource=0; }
   }
  //----------------------------------------------------------------



   if (tX==0) { } else
   if (tX<0) { bitbltColorRGB(target,targetWidth+tX,0,targetWidth,targetHeight,transR,transG,transB,abs(tX),targetHeight); } else
             { bitbltColorRGB(target,0,0,targetWidth,targetHeight,transR,transG,transB,abs(tX),targetHeight); }

   if (tY==0) { } else
   if (tY<0) { bitbltColorRGB(target,0,targetHeight+tY,targetWidth,targetHeight,transR,transG,transB,targetWidth,abs(tY)); } else
             { bitbltColorRGB(target,0,0,targetWidth,targetHeight,transR,transG,transB,targetWidth,abs(tY)); }

return 1;

}





int shiftImageDepth(unsigned short * target, unsigned short * source , unsigned short depthVal , signed int tX,  signed int tY  ,  unsigned int width , unsigned int height)
{
  if ( (target==0)||(source==0) ) { return 0; }
  if ( (width==0)&&(height==0) ) { return 0; }


  unsigned int sourceWidth=width,sourceHeight=height,targetWidth=width,targetHeight=height;
  unsigned int sourceX,sourceY , targetX,targetY;

  if (tX < 0 ) {   sourceX=abs(tX);    targetX=0;   } else
               {   sourceX=0;          targetX=abs(tX); }

  if (tY < 0 ) { sourceY=abs(tY); targetY=0;  } else
               { sourceY=0;       targetY=abs(tY); }

  width=width - abs(tX)-1;
  height=height - abs(tY)-1;

  if (width>sourceWidth) { width=sourceWidth; fprintf(stderr,"Error setting width (?) why did this happen ? :P \n"); }
  if (height>sourceHeight) { height=sourceHeight; fprintf(stderr,"Error setting height (?) why did this happen ? :P \n"); }

  fprintf(stderr,"Doing shift (%d,%d) by bit blitting %u,%u to %u,%u ( size %u,%u) \n",tX,tY,sourceX,sourceY,targetX,targetY,width,height);

  //----------------------------------------------------------------
  unsigned short * maybeCopiedSource = source;
  //In case tx or ty is positive it is impossible to bit blt using the same buffer since lines will be overriden so we use a seperate buffer
  if ( (tX>0) || (tY>0) )
  {
    unsigned int copySize = width*height*1*sizeof(unsigned short);
    maybeCopiedSource = (unsigned short * ) malloc(copySize);
    if (maybeCopiedSource==0) { maybeCopiedSource = source; } else
                              { memcpy(maybeCopiedSource,source,copySize); }
  }
  //----------------------------------------------------------------

        bitbltDepth( target ,targetX,targetY, targetWidth,targetHeight,
                     maybeCopiedSource ,sourceX,sourceY, sourceWidth,sourceHeight,
                     width,height);


  //----------------------------------------------------------------
   if ( (tX>0) || (tY>0) )
   {
     if (maybeCopiedSource!=source) { free(maybeCopiedSource); maybeCopiedSource=0; }
   }
  //----------------------------------------------------------------

   if (tX==0) { } else
   if (tX<0)  { bitbltDepthValue(target,targetWidth+tX,0,targetWidth,targetHeight,depthVal,abs(tX),targetHeight); } else
              { bitbltDepthValue(target,0,0,targetWidth,targetHeight,depthVal,abs(tX),targetHeight); }

   if (tY==0) { } else
   if (tY<0)  { bitbltDepthValue(target,0,targetHeight+tY,targetWidth,targetHeight,depthVal,targetWidth,abs(tY)); } else
              { bitbltDepthValue(target,0,0,targetWidth,targetHeight,depthVal,targetWidth,abs(tY)); }

  return 1;
}




int mixbltRGB(unsigned char * target,  unsigned int tX,  unsigned int tY , unsigned int targetWidth , unsigned int targetHeight ,
              unsigned char * source , unsigned int sX, unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
              unsigned int width , unsigned int height)
{
  if ( (target==0)||(source==0) ) { return 0; }
  if ( (width==0)&&(height==0) ) { return 0; }
  if ( (sourceWidth==0)&&(sourceHeight==0) ) { return 0; }

  //Check for bounds -----------------------------------------
  if (tX+width>=targetWidth) { width=targetWidth-tX-1;  }
  if (tY+height>=targetHeight) { height=targetHeight-tY-1;  }

  if (sX+width>=sourceWidth) { width=sourceWidth-sX-1;  }
  if (sY+height>=sourceHeight) { height=sourceHeight-sY-1;  }
  //----------------------------------------------------------

  unsigned char * sourcePTR; unsigned char * sourceLineLimitPTR; unsigned char * sourceLimitPTR; unsigned int sourceLineSkip;
  unsigned char * targetPTR;  /*unsigned char * targetLimitPTR;*/  unsigned int targetLineSkip;


  sourcePTR      = source+ MEMPLACE3(sX,sY,sourceWidth);
  sourceLimitPTR = source+ MEMPLACE3((sX+width),(sY+height),sourceWidth);
  sourceLineSkip = (sourceWidth-width) * 3;
  sourceLineLimitPTR = sourcePTR + (width*3);
  fprintf(stderr,"SOURCE (RGB %u/%u)  Starts at %u,%u and ends at %u,%u\n",sourceWidth,sourceHeight,sX,sY,sX+width,sY+height);

  targetPTR      = target + MEMPLACE3(tX,tY,targetWidth);
  //targetLimitPTR = target + MEMPLACE3((tX+width),(tY+height),targetWidth);
  targetLineSkip = (targetWidth-width) * 3;
  fprintf(stderr,"TARGET (RGB %u/%u)  Starts at %u,%u and ends at %u,%u\n",targetWidth,targetHeight,tX,tY,tX+width,tY+height);

  unsigned int color = 0;

  while (sourcePTR < sourceLimitPTR)
  {
     while (sourcePTR < sourceLineLimitPTR)
     {
        if (*sourcePTR==0) { color = *targetPTR; } else { color = (unsigned int) ( *targetPTR + *sourcePTR ) / 2; }
        *targetPTR =  (unsigned char ) color;
        ++targetPTR; ++sourcePTR;

        if (*sourcePTR==0) { color = *targetPTR; } else { color = (unsigned int) ( *targetPTR + *sourcePTR ) / 2; }
        *targetPTR =  (unsigned char ) color;
        ++targetPTR; ++sourcePTR;

        if (*sourcePTR==0) { color = *targetPTR; } else { color = (unsigned int) ( *targetPTR + *sourcePTR ) / 2; }
        *targetPTR =  (unsigned char ) color;
        ++targetPTR; ++sourcePTR;
     }


    sourceLineLimitPTR+= sourceWidth*3;//*3;
    targetPTR+=targetLineSkip;
    sourcePTR+=sourceLineSkip;
  }
 return 1;
}






int bitbltRGB(unsigned char * target,  unsigned int tX,  unsigned int tY , unsigned int targetWidth , unsigned int targetHeight ,
              unsigned char * source , unsigned int sX, unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
              unsigned int width , unsigned int height)
{
  if ( (target==0)||(source==0) ) { return 0; }
  if ( (width==0)&&(height==0) ) { return 0; }
  if ( (sourceWidth==0)&&(sourceHeight==0) ) { return 0; }

  fprintf(stderr,"BitBlt an area of target image %u,%u  sized %u,%u \n",tX,tY,targetWidth,targetHeight);
  fprintf(stderr,"BitBlt an area of source image %u,%u  sized %u,%u \n",sX,sY,sourceWidth,sourceHeight);
  fprintf(stderr,"BitBlt size was width %u height %u \n",width,height);
  //Check for bounds -----------------------------------------
  if (tX+width>=targetWidth) { width=targetWidth-tX-1;  }
  if (tY+height>=targetHeight) { height=targetHeight-tY-1;  }

  if (sX+width>=sourceWidth) { width=sourceWidth-sX-1;  }
  if (sY+height>=sourceHeight) { height=sourceHeight-sY-1;  }
  //----------------------------------------------------------
  fprintf(stderr,"BitBlt size NOW is width %u height %u \n",width,height);

  unsigned char *  sourcePTR      = source+ MEMPLACE3(sX,sY,sourceWidth);
  unsigned char *  sourceLimitPTR = source+ MEMPLACE3((sX+width),(sY+height),sourceWidth);
  unsigned int     sourceLineSkip = (sourceWidth-width) * 3;
  unsigned char *  sourceLineLimitPTR = sourcePTR + (width*3) -3; /*-3 is required here*/
  //fprintf(stderr,"SOURCE (RGB size %u/%u)  Starts at %u,%u and ends at %u,%u\n",sourceWidth,sourceHeight,sX,sY,sX+width,sY+height);
  //fprintf(stderr,"sourcePTR is %p , limit is %p \n",sourcePTR,sourceLimitPTR);
  //fprintf(stderr,"sourceLineSkip is %u\n",        sourceLineSkip);
  //fprintf(stderr,"sourceLineLimitPTR is %p\n",sourceLineLimitPTR);


  unsigned char * targetPTR      = target + MEMPLACE3(tX,tY,targetWidth);
  unsigned char * targetLimitPTR = target + MEMPLACE3((tX+width),(tY+height),targetWidth);
  unsigned int targetLineSkip = (targetWidth-width) * 3;
  unsigned char * targetLineLimitPTR = targetPTR + (width*3) -3; /*-3 is required here*/
  //fprintf(stderr,"TARGET (RGB size %u/%u)  Starts at %u,%u and ends at %u,%u\n",targetWidth,targetHeight,tX,tY,tX+width,tY+height);
  //fprintf(stderr,"targetPTR is %p , limit is %p \n",targetPTR,targetLimitPTR);
  //fprintf(stderr,"targetLineSkip is %u\n", targetLineSkip);
  //fprintf(stderr,"targetLineLimitPTR is %p\n",targetLineLimitPTR);

  while ( (sourcePTR < sourceLimitPTR) && ( targetPTR+3 < targetLimitPTR ) )
  {
     while ( (sourcePTR < sourceLineLimitPTR) && ((targetPTR+3 < targetLineLimitPTR)) )
     {
        //fprintf(stderr,"Reading Triplet sourcePTR %p targetPTR is %p\n",sourcePTR  ,targetPTR);
        *targetPTR = *sourcePTR; ++targetPTR; ++sourcePTR;
        *targetPTR = *sourcePTR; ++targetPTR; ++sourcePTR;
        *targetPTR = *sourcePTR; ++targetPTR; ++sourcePTR;
     }

    sourceLineLimitPTR += sourceWidth*3;
    targetLineLimitPTR += targetWidth*3;
    sourcePTR+=sourceLineSkip;
    targetPTR+=targetLineSkip;
  }

 return 1;
}





int bitbltColorRGB(unsigned char * target,  unsigned int tX,  unsigned int tY  , unsigned int targetWidth , unsigned int targetHeight ,
                   unsigned char R , unsigned char G , unsigned char B ,
                   unsigned int width , unsigned int height)
{
  //Check for bounds -----------------------------------------
  if (tX+width>=targetWidth) { width=targetWidth-tX-1;  }
  if (tY+height>=targetHeight) { height=targetHeight-tY-1;  }
  //----------------------------------------------------------

  unsigned char * targetPTR; unsigned char * targetLineLimitPTR; unsigned char * targetLimitPTR;   unsigned int targetLineSkip;
  targetPTR      = target + MEMPLACE3(tX,tY,targetWidth);
  targetLimitPTR = target + MEMPLACE3((tX+width),(tY+height),targetWidth);
  targetLineSkip = (targetWidth-width) * 3;
  targetLineLimitPTR = targetPTR + (width*3) -3; /*-3 is required here*/

  fprintf(stderr,"BitBlt Color an area (%u,%u) of target image  starting at %u,%u  sized %u,%u with color RGB(%u,%u,%u)\n",width,height,tX,tY,targetWidth,targetHeight,R,G,B);
  fprintf(stderr,"last Pixels @ %u,%u\n",tX+width,tY+height);
  while ( targetPTR < targetLimitPTR )
  {
     while (targetPTR < targetLineLimitPTR)
     {
        //fprintf(stderr,"Reading Triplet sourcePTR %p targetPTR is %p\n",sourcePTR  ,targetPTR);
        *targetPTR = R; ++targetPTR;
        *targetPTR = G; ++targetPTR;
        *targetPTR = B; ++targetPTR;
     }
    targetLineLimitPTR += targetWidth*3;
    targetPTR+=targetLineSkip;
  }
 return 1;
}



int bitbltDepthValue(unsigned short * target,  unsigned int tX,  unsigned int tY  , unsigned int targetWidth , unsigned int targetHeight ,
                     unsigned short DepthVal ,
                     unsigned int width , unsigned int height)
{
  //Check for bounds -----------------------------------------
  if (tX+width>=targetWidth) { width=targetWidth-tX-1;  }
  if (tY+height>=targetHeight) { height=targetHeight-tY-1;  }
  //----------------------------------------------------------

  unsigned short * targetPTR; unsigned short * targetLineLimitPTR; unsigned short * targetLimitPTR;   unsigned int targetLineSkip;
  targetPTR      = target + MEMPLACE1(tX,tY,targetWidth);
  targetLimitPTR = target + MEMPLACE1((tX+width),(tY+height),targetWidth);
  targetLineSkip = (targetWidth-width);
  targetLineLimitPTR = targetPTR + (width) -1 ;

  fprintf(stderr,"BitBlt Depth an area (%u,%u) of target image  starting at %u,%u  sized %u,%u with Depth(%u)\n",width,height,tX,tY,targetWidth,targetHeight,DepthVal);
  while ( targetPTR < targetLimitPTR )
  {
     while (targetPTR < targetLineLimitPTR)
     {
        //fprintf(stderr,"Reading Triplet sourcePTR %p targetPTR is %p\n",sourcePTR  ,targetPTR);
        *targetPTR = DepthVal; ++targetPTR;
     }
    targetLineLimitPTR += targetWidth;
    targetPTR+=targetLineSkip;
  }
 return 1;
}



int bitbltDepth(unsigned short * target,  unsigned int tX,  unsigned int tY  , unsigned int targetWidth , unsigned int targetHeight ,
                unsigned short * source , unsigned int sX,  unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
                unsigned int width , unsigned int height)
{
  if ( (target==0)||(source==0) ) { return 0; }
  if ( (width==0)&&(height==0) ) { return 0; }
  if ( (sourceWidth==0)&&(sourceHeight==0) ) { return 0; }

  //Check for bounds -----------------------------------------
  if (tX+width>=targetWidth) { width=targetWidth-tX-1;  }
  if (tY+height>=targetHeight) { height=targetHeight-tY-1;  }

  if (sX+width>=sourceWidth) { width=sourceWidth-sX-1;  }
  if (sY+height>=sourceHeight) { height=sourceHeight-sY-1;  }
  //----------------------------------------------------------

  unsigned short * sourcePTR;  unsigned short* sourceLineLimitPTR; unsigned short * sourceLimitPTR; unsigned int sourceLineSkip;
  unsigned short * targetPTR;  /*unsigned short * targetLimitPTR;*/    unsigned int targetLineSkip;


  sourcePTR      = source+ MEMPLACE1(sX,sY,sourceWidth);
  sourceLimitPTR = source+ MEMPLACE1((sX+width),(sY+height),sourceWidth);
  sourceLineSkip = (sourceWidth-width)  ;
  sourceLineLimitPTR = sourcePTR + (width) -1;
  //fprintf(stderr,"SOURCE (Depth %u/%u)  Starts at %u,%u and ends at %u,%u\n",sourceWidth,sourceHeight,sX,sY,sX+width,sY+height);

  targetPTR      = target + MEMPLACE1(tX,tY,targetWidth);
  //targetLimitPTR = target + MEMPLACE1((tX+width),(tY+height),targetWidth);
  targetLineSkip = (targetWidth-width)  ;
  //fprintf(stderr,"TARGET (Depth %u/%u)  Starts at %u,%u and ends at %u,%u\n",targetWidth,targetHeight,tX,tY,tX+width,tY+height);

  while (sourcePTR < sourceLimitPTR)
  {
     while (sourcePTR < sourceLineLimitPTR)
     {
        *targetPTR =  *sourcePTR;
        ++targetPTR; ++sourcePTR;
     }

    sourceLineLimitPTR+= sourceWidth;
    targetPTR+=targetLineSkip;
    sourcePTR+=sourceLineSkip;
  }
 return 1;
}





int printOutHistogram(char * filename, unsigned int * RHistogram_1 , unsigned int * GHistogram_1 , unsigned int * BHistogram_1 , unsigned int Samples_1  )
{
  unsigned int i=0;

  FILE *fpr = 0;

  char filenameInt[255];
  sprintf(filenameInt,"RED%s",filename);
  fpr=fopen(filenameInt,"w");
  if (fpr!=0)
  {
   for (i=0; i<256; i++) { fprintf(fpr,"%u\n",RHistogram_1[i]); }
   fclose(fpr);
  }


  sprintf(filenameInt,"GREEN%s",filename);
  fpr=fopen(filenameInt,"w");
  if (fpr!=0)
  {
   for (i=0; i<256; i++) { fprintf(fpr,"%u\n",GHistogram_1[i]); }
   fclose(fpr);
  }


  sprintf(filenameInt,"BLUE%s",filename);
  fpr=fopen(filenameInt,"w");
  if (fpr!=0)
  {
   for (i=0; i<256; i++) { fprintf(fpr,"%u\n",BHistogram_1[i]); }
   fclose(fpr);
  }

  char command[1024]={0};
  sprintf(command,"gnuplot -e 'set terminal png; set output \"RED%s.png\"; set title \"3D random points\"; plot \"RED%s\" with lines'",filename,filename);
  i=system(command);

  sprintf(command,"gnuplot -e 'set terminal png; set output \"GREEN%s.png\"; set title \"3D random points\"; plot \"GREEN%s\" with lines'",filename,filename);
  i=system(command);

  sprintf(command,"gnuplot -e 'set terminal png; set output \"BLUE%s.png\"; set title \"3D random points\"; plot \"BLUE%s\" with lines'",filename,filename);
  i=system(command);
 return 1;
}

unsigned int compareHistogram(unsigned int * RHistogram , unsigned int * GHistogram , unsigned int * BHistogram , unsigned int * samples ,
                     unsigned int * minRHistogram , unsigned int * minGHistogram , unsigned int * minBHistogram ,
                     unsigned int * maxRHistogram , unsigned int * maxGHistogram , unsigned int * maxBHistogram  )
{
  unsigned int totalRDiff = 0;
  unsigned int totalGDiff = 0;
  unsigned int totalBDiff = 0;
  unsigned int i=0;

  for (i=0; i<256; i++)
  {
    if (minRHistogram[i]>RHistogram[i])  { totalRDiff+= minRHistogram[i]-RHistogram[i]; } else
    if (maxRHistogram[i]<RHistogram[i])  { totalRDiff+= RHistogram[i]-maxRHistogram[i]; }

    if (minGHistogram[i]>GHistogram[i])  { totalGDiff+= minGHistogram[i]-GHistogram[i]; } else
    if (maxGHistogram[i]<GHistogram[i])  { totalGDiff+= GHistogram[i]-maxGHistogram[i]; }

    if (minBHistogram[i]>BHistogram[i])  { totalBDiff+= minBHistogram[i]-BHistogram[i]; } else
    if (maxBHistogram[i]<BHistogram[i])  { totalBDiff+= BHistogram[i]-maxBHistogram[i]; }
  }

 return totalRDiff+totalBDiff+totalGDiff;
}



int updateHistogramFilter(
                           unsigned int * RHistogram , unsigned int * GHistogram , unsigned int * BHistogram , unsigned int * samples ,
                           unsigned int * minRHistogram , unsigned int * minGHistogram , unsigned int * minBHistogram   ,
                           unsigned int * maxRHistogram , unsigned int * maxGHistogram , unsigned int * maxBHistogram
                         )
{
  unsigned int i=0;
  for (i=0; i<256; i++)
  {
    if (minRHistogram[i]>RHistogram[i])  { minRHistogram[i]=RHistogram[i]; }
    if (minGHistogram[i]>GHistogram[i])  { minGHistogram[i]=GHistogram[i]; }
    if (minBHistogram[i]>BHistogram[i])  { minBHistogram[i]=BHistogram[i]; }

    if (maxRHistogram[i]<RHistogram[i])  { maxRHistogram[i]=RHistogram[i]; }
    if (maxGHistogram[i]<GHistogram[i])  { maxGHistogram[i]=GHistogram[i]; }
    if (maxBHistogram[i]<BHistogram[i])  { maxBHistogram[i]=BHistogram[i]; }
  }
 return 1;
}



int saveHistogramFilter(
                           char * filename ,
                           unsigned int * minRHistogram , unsigned int * minGHistogram , unsigned int * minBHistogram   ,
                           unsigned int * maxRHistogram , unsigned int * maxGHistogram , unsigned int * maxBHistogram
                         )
{
  FILE *fpr = 0;

  fpr=fopen(filename,"w");
  if (fpr!=0)
  {
   unsigned int i=0;


   fprintf(fpr,"unsigned int minRHistogram[256]={");
   for (i=0; i<255; i++) { fprintf(fpr,"%u,",minRHistogram[i]); }
   fprintf(fpr,"%u};\n ",minBHistogram[255]);

   fprintf(fpr,"unsigned int minGHistogram[256]={");
   for (i=0; i<255; i++) { fprintf(fpr,"%u,",minGHistogram[i]); }
   fprintf(fpr,"%u};\n ",minBHistogram[255]);

   fprintf(fpr,"unsigned int minBHistogram[256]={");
   for (i=0; i<255; i++) { fprintf(fpr,"%u,",minBHistogram[i]); }
   fprintf(fpr,"%u};\n\n ",minBHistogram[255]);


   fprintf(fpr,"unsigned int maxRHistogram[256]={");
   for (i=0; i<255; i++) { fprintf(fpr,"%u,",maxRHistogram[i]); }
   fprintf(fpr,"%u};\n ",maxBHistogram[255]);

   fprintf(fpr,"unsigned int maxGHistogram[256]={");
   for (i=0; i<255; i++) { fprintf(fpr,"%u,",maxGHistogram[i]); }
   fprintf(fpr,"%u};\n ",maxBHistogram[255]);

   fprintf(fpr,"unsigned int maxBHistogram[256]={");
   for (i=0; i<255; i++) { fprintf(fpr,"%u,",maxBHistogram[i]); }
   fprintf(fpr,"%u};\n\n ",maxBHistogram[255]);

   fprintf(fpr,"void initHistogramLimits()\n");
   fprintf(fpr,"{ return; //AUTOMATICALLY DISABLED \n");
   fprintf(fpr," unsigned int i=0;\n");
   fprintf(fpr," for (i=0; i<256; i++)\n");
   fprintf(fpr," {\n");
   fprintf(fpr,"   minRHistogram[i]=10000;");
   fprintf(fpr,"   minGHistogram[i]=10000;");
   fprintf(fpr,"   minBHistogram[i]=10000;");
   fprintf(fpr," }\n");
   fprintf(fpr,"}\n");

   fclose(fpr);
  }

 return 1;
}



int calculateHistogram(unsigned char * target,  unsigned int tX,  unsigned int tY  , unsigned int targetWidth , unsigned int targetHeight ,
                       unsigned int * RHistogram , unsigned int * GHistogram , unsigned int * BHistogram , unsigned int * samples ,
                       unsigned int width , unsigned int height)
{
  if ( (RHistogram==0)||(GHistogram==0)||(BHistogram==0) )
  {
      fprintf(stderr,"Cannot Calculate Histogram without a target histogram output");
      return 0;
  }

  *samples=0;
  memset(RHistogram,0,255);
  memset(GHistogram,0,255);
  memset(BHistogram,0,255);

  fprintf(stderr,"Initially a Histogram at an area (%u,%u) of target image  starting at %u,%u  sized %u,%u  \n",width,height,tX,tY,targetWidth,targetHeight);
  //Check for bounds -----------------------------------------
  if (tX+width>=targetWidth) { width=targetWidth-tX-1;  }
  if (tY+height>=targetHeight) { height=targetHeight-tY-1;  }
  //----------------------------------------------------------

  unsigned char * targetPTR; unsigned char * targetLineLimitPTR; unsigned char * targetLimitPTR;   unsigned int targetLineSkip;
  targetPTR      = target + MEMPLACE3(tX,tY,targetWidth);
  targetLimitPTR = target + MEMPLACE3((tX+width),(tY+height),targetWidth);
  targetLineSkip = (targetWidth-width) * 3;
  targetLineLimitPTR = targetPTR + (width*3) -3; /*-3 is required here*/

  fprintf(stderr,"Calculating a Histogram at an area (%u,%u) of target image  starting at %u,%u  sized %u,%u  \n",width,height,tX,tY,targetWidth,targetHeight);

  while ( targetPTR < targetLimitPTR )
  {
     while (targetPTR < targetLineLimitPTR)
     {
        //fprintf(stderr,"Reading Triplet sourcePTR %p targetPTR is %p\n",sourcePTR  ,targetPTR);
        RHistogram[*targetPTR]+=1; ++targetPTR;
        GHistogram[*targetPTR]+=1; ++targetPTR;
        BHistogram[*targetPTR]+=1; ++targetPTR;
     }
    targetLineLimitPTR += targetWidth*3;
    targetPTR+=targetLineSkip;
  }

  fprintf(stderr,"Done\n");

  *samples=targetWidth*targetHeight;


 return 1;
}


int saveTileRGBToFile(  unsigned int solutionColumn , unsigned int solutionRow ,
                        unsigned char * source , unsigned int sX, unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
                        unsigned int width , unsigned int height)
{

 char filename[512];
 sprintf(filename,"tiles/rgb_tile%u_%u.pnm",solutionColumn,solutionRow);


 unsigned char * tile = (unsigned char*) malloc((width+1)*(height+1)*3*sizeof(unsigned char));
 if (tile == 0 ) { return 0; }
 bitbltRGB(tile,0,0,width,height,source,sX,sY,sourceWidth,sourceHeight,width, height);


 saveRawImageToFile(filename ,0,tile , width , height, 3 , 8);
 free(tile);

 return 1;
}

int saveTileDepthToFile(  unsigned int solutionColumn , unsigned int solutionRow ,
                          unsigned short * source , unsigned int sX, unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
                          unsigned int width , unsigned int height)
{

 char filename[512];
 sprintf(filename,"tiles/depth_tile%u_%u.pnm",solutionColumn,solutionRow);


 unsigned short * tile = (unsigned short*) malloc((width+1)*(height+1)*1*sizeof(unsigned short));
 if (tile == 0 ) { return 0; }
 bitbltDepth(tile,0,0,width,height,source,sX,sY,sourceWidth,sourceHeight,width, height);


 saveRawImageToFile(filename ,0,(unsigned char*) tile , width , height, 1 , 16);
 free(tile);

 return 1;
}








int bitBltRGBToFile(  char * name  , char * comment ,
                      unsigned char * source , unsigned int sX, unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
                      unsigned int width , unsigned int height)
{

 char filename[512];
 sprintf(filename,"%s.pnm",name);


 unsigned char * tile = (unsigned char*) malloc((width+1)*(height+1)*3*sizeof(unsigned char));
 if (tile == 0 ) { return 0; }
 bitbltRGB(tile,0,0,width,height,source,sX,sY,sourceWidth,sourceHeight,width, height);


 if ( ! saveRawImageToFile(filename ,comment,tile , width , height, 3 , 8) )
 {
     fprintf(stderr,"Could not bit blt to File %s\n",name);
 }
 free(tile);

 return 1;
}




int bitBltDepthToFile(  char * name  ,char * comment ,
                        unsigned short * source , unsigned int sX, unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
                        unsigned int width , unsigned int height)
{

 char filename[512];
 sprintf(filename,"%s.pnm",name);


 unsigned short * tile = (unsigned short*) malloc((width+1)*(height+1)*1*sizeof(unsigned short));
 if (tile == 0 ) { return 0; }
 bitbltDepth(tile,0,0,width,height,source,sX,sY,sourceWidth,sourceHeight,width, height);


 saveRawImageToFile(filename ,comment,(unsigned char*) tile , width , height, 1  , 16);
 free(tile);

 return 1;
}



unsigned int countOccurancesOfRGBPixel(unsigned char * ptrRGB , unsigned int RGBwidth , unsigned int RGBheight , unsigned char transR ,unsigned char transG , unsigned char transB)
{
 unsigned int cCount = 0;
 unsigned char * sourcePTR =  ptrRGB ;
 unsigned char * sourceLimitPTR =  ptrRGB + (RGBwidth*RGBheight *3);
 unsigned char R,G,B;

  while (sourcePTR < sourceLimitPTR)
  {
    R = *sourcePTR; ++sourcePTR;
    G = *sourcePTR; ++sourcePTR;
    B = *sourcePTR; ++sourcePTR;
    if ( (R==transR) && (G==transG) && (B==transB) ) { ++cCount; }
  }

 return cCount;
}



int getRGBPixel(unsigned char * ptrRGB , unsigned int RGBwidth , unsigned int RGBheight ,  unsigned int x , unsigned int y , unsigned char * R , unsigned char * G , unsigned char * B)
{
 unsigned char * ptr =  ptrRGB  + MEMPLACE3(x,y,RGBwidth);

 *R = *ptr; ++ptr;
 *G = *ptr; ++ptr;
 *B = *ptr; ++ptr;

 return 1;
}




unsigned short getDepthPixel(unsigned short * ptrDepth , unsigned int Depthwidth , unsigned int Depthheight ,  unsigned int x , unsigned int y)
{
 unsigned short * ptr =  ptrDepth  + MEMPLACE1(x,y,Depthwidth);
 return *ptr;
}



int setDepthPixel(unsigned short * ptrDepth , unsigned int Depthwidth , unsigned int Depthheight ,  unsigned int x , unsigned int y , unsigned short depthValue)
{
 unsigned short * ptr =  ptrDepth  + MEMPLACE1(x,y,Depthwidth);
 *ptr = depthValue;
 return 1;
}



int closeToRGB(unsigned char R , unsigned char G , unsigned char B  ,  unsigned char targetR , unsigned char targetG , unsigned char targetB , unsigned int threshold)
{
 if ( ( ABSDIFF(R,targetR) < threshold ) && ( ABSDIFF(G,targetG) < threshold ) && ( ABSDIFF(B,targetB) < threshold ) )   { return 1; }
 return 0;
}




unsigned int countDepthAverage(unsigned short * source, unsigned int sourceWidth , unsigned int sourceHeight ,
                                unsigned int sX,  unsigned int sY  , unsigned int tileWidth , unsigned int tileHeight)
{
  //Check for bounds -----------------------------------------
  if (sX+tileWidth>=sourceWidth) { tileWidth=sourceWidth-sX-1;  }
  if (sY+tileHeight>=sourceHeight) { tileHeight=sourceHeight-sY-1;  }
  //----------------------------------------------------------

  unsigned short * sourcePTR; unsigned short * sourceLineLimitPTR; unsigned short * sourceLimitPTR;   unsigned int sourceLineSkip;
  sourcePTR      = source + MEMPLACE1(sX,sY,sourceWidth);
  sourceLimitPTR = source + MEMPLACE1((sX+tileWidth),(sY+tileHeight),sourceWidth);
  sourceLineSkip = (sourceWidth-tileWidth);
  sourceLineLimitPTR = sourcePTR + (tileWidth) -1 ;

  //fprintf(stderr,"Getting Average Depth at area (%u,%u) of source image  starting at %u,%u  sized %u,%u \n",tileWidth,tileHeight,sX,sY,sourceWidth,sourceHeight);
  unsigned int curDepth = 0;
  unsigned int totalDepth = 0;
  unsigned int totalMeasurements = 0;

  while ( sourcePTR < sourceLimitPTR )
  {
     while (sourcePTR < sourceLineLimitPTR)
     {
        //fprintf(stderr,"Reading Triplet sourcePTR %p targetPTR is %p\n",sourcePTR  ,targetPTR);
        if (*sourcePTR!=0)
             {
               curDepth = (unsigned int) *sourcePTR;
               totalDepth += curDepth;
               ++totalMeasurements;
             }
        ++sourcePTR;
     }
    sourceLineLimitPTR += sourceWidth;
    sourcePTR+=sourceLineSkip;
  }
 //fprintf(stderr,"Initial total is %u after %u measurments \n",totalDepth,totalMeasurements);

 if (totalMeasurements==0) { return 0; }
 return (unsigned int) (totalDepth / totalMeasurements);
}


