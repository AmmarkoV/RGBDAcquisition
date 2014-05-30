#include <stdio.h>
#include <math.h>
#include "imageProcessing.h"

#define PI (3.141592653589793)
#define MEMPLACE1(x,y,width) ( y * ( width  ) + x )

enum dimEnum
{
    DIMX = 0 ,
    DIMY ,
    DIMZ ,
    NUMBER_OF_DIMENSIONS
};



int normalizeNormal(float * normal)
{
 float tempLength =(normal[DIMX]*normal[DIMX]) + (normal[DIMY]*normal[DIMY]) + (normal[DIMZ]*normal[DIMZ]);

 if (tempLength>0)
  {
   tempLength = sqrt(tempLength);

   normal[DIMX] /= tempLength;
   normal[DIMY] /= tempLength;
   normal[DIMZ] /= tempLength;
   return 1;
  } else
  {
    fprintf(stderr,"Error normalizing normal ( %f , %f , %f ) \n",normal[0],normal[1],normal[2]);
  }
 return 0;
}


void crossProductFrom3Points(float * p1 , float * p2 , float * p3  , float * normal)
{
 // fprintf(stderr,"Point 1 %0.5f %0.5f %0.5f \n",p1[0],p1[1],p1[2]);
 // fprintf(stderr,"Point 2 %0.5f %0.5f %0.5f \n",p2[0],p2[1],p2[2]);
 // fprintf(stderr,"Point 3 %0.5f %0.5f %0.5f \n",p3[0],p3[1],p3[2]);

 float temp_v1[3];
 float temp_v2[3];

 int i=0;
 for (i=0; i<3; i++)
 {
   temp_v1[i]=p1[i]-p2[i];
   temp_v2[i]=p3[i]-p2[i];
 }

 // calculate cross product
 normal[DIMX] = temp_v1[DIMY]*temp_v2[DIMZ] - temp_v1[DIMZ]*temp_v2[DIMY];
 normal[DIMY] = temp_v1[DIMZ]*temp_v2[DIMX] - temp_v1[DIMX]*temp_v2[DIMZ];
 normal[DIMZ] = temp_v1[DIMX]*temp_v2[DIMY] - temp_v1[DIMY]*temp_v2[DIMX];


 // fprintf(stderr,"Cross Product is %0.2f %0.2f %0.2f \n",normal[0],normal[1],normal[2]);
 normalizeNormal(normal);
 // fprintf(stderr,"Cross Product Normalized is %0.2f %0.2f %0.2f \n",normal[0],normal[1],normal[2]);

}

float magnitudeOfNormal(float * p1)
{
  float tmp = ((float) p1[0]*p1[0]) + ((float)p1[1]*p1[1]) + ((float)p1[2]*p1[2]);

  if (tmp>=0.0)
  {
    return sqrt(tmp);
  }

  fprintf(stderr,"Error Calculating Magnitude of Normal\n");
  return 0.0;
}


float  angleOfNormals(float * p1 , float * p2)
{
   float mag1 =  magnitudeOfNormal(p1);
   float mag2 =  magnitudeOfNormal(p2);

   float denominator = mag1 * mag2;

   if (denominator==0.0) {
                           //fprintf(stderr,"Error calculating angleOfNormals between (%f %f %f ) and (%f %f %f ) \n",p1[0],p1[1],p1[2],p2[0],p2[1],p2[2]);
                           //fprintf(stderr,"  ( magnitudes %f %f produce a zero denominator  )\n",mag1,mag2);
                           return NAN; }

   float numerator = innerProduct(p1,p2);

   float result = acos((float) numerator / denominator ) * 180.0 / PI;
   return result;
}


float  distance3D(float * p1 , float * p2 , float * p3)
{
   float vect_x = p1[DIMX] - p2[DIMX];
   float vect_y = p1[DIMY] - p2[DIMY];
   float vect_z = p1[DIMZ] - p2[DIMZ];

   float len = sqrt( vect_x*vect_x + vect_y*vect_y + vect_z*vect_z);
   if(len == 0) len = 1.0f;

   return len;
}


float dotProduct(float * p1 , float * p2 )
{
    #warning "dotProduct is a very heavily used function , it needs to be optimized using AVX"
    return (float) ( p1[DIMX]*p2[DIMX] + p1[DIMY]*p2[DIMY] + p1[DIMZ]*p2[DIMZ] );
}


float  signedDistanceFromPlane(float * origin , float * normal , float * pN)
{
  #warning "signedDistanceFromPlane is a very heavily used function , it needs to be optimized using AVX"
  float tempV[NUMBER_OF_DIMENSIONS];
  int i=0;
  for (i=0; i<NUMBER_OF_DIMENSIONS; i++)
  {
      tempV[i]=pN[i]-origin[i];
  }

  return dotProduct(tempV,normal);
}



int getDepthBlobAverage(unsigned short * frame , unsigned int frameWidth , unsigned int frameHeight,
                        unsigned int sX,unsigned int sY,unsigned int width,unsigned int height,
                        float * centerX , float * centerY , float * centerZ)
{

  if (frame==0)  { return 0; }
  if ( (width==0)||(height==0) ) { return 0; }
  if ( (frameWidth==0)||(frameWidth==0) ) { return 0; }

  if (sX>=frameWidth) { return 0; }
  if (sY>=frameHeight) { return 0;  }

  //Check for bounds -----------------------------------------
  if (sX+width>=frameWidth) { width=frameWidth-sX;  }
  if (sY+height>=frameHeight) { height=frameHeight-sY;  }
  //----------------------------------------------------------


  unsigned int x=0,y=0;
  unsigned long sumX=0,sumY=0,sumZ=0,samples=0;

  unsigned short * sourcePTR      = frame+ MEMPLACE1(sX,sY,frameWidth);
  unsigned short * sourceLimitPTR = frame+ MEMPLACE1((sX+width),(sY+height),frameWidth);
  unsigned short sourceLineSkip = (frameWidth-width)  ;
  unsigned short * sourceLineLimitPTR = sourcePTR + (width);

  while (sourcePTR < sourceLimitPTR)
  {
     while (sourcePTR < sourceLineLimitPTR)
     {
       if (*sourcePTR!=0)
       {
        sumX+=x;
        sumY+=y;
        sumZ+=*sourcePTR;
        ++samples;
       }

       ++x;
       ++sourcePTR;
     }

    x=0; ++y;
    sourceLineLimitPTR+=frameWidth;
    sourcePTR+=sourceLineSkip;
  }


   *centerX = (float) sumX / samples;
   *centerY = (float) sumY / samples;
   *centerZ = (float) sumZ / samples;
   return 1;
}





int floodFill(unsigned char * target , unsigned int width , unsigned int height ,
                signed int pX , signed int pY , int threshold,
                unsigned char sR , unsigned char sG , unsigned char sB ,
                unsigned char R , unsigned char G , unsigned char B , int depth)
{
 if ( (pX<0) || (pY<0) || (pX>=width) || (pY>=height) ) { return 0; }
 if (depth>2000) { return 0; }

 if (target==0) { return 0; }
 if (width==0) { return 0; }
 if (height==0) { return 0; }

 unsigned char * source = (unsigned char *) target  + ( (pX*3) + pY * width*3 );

 unsigned char * tR = source; ++source;
 unsigned char * tG = source; ++source;
 unsigned char * tB = source;


  if ( (depth>0) && ( *tR == R   ) &&  ( *tG == G   )  &&  ( *tB == B ) ) { return 0; }


  if (
       (( *tR > sR-threshold ) && ( *tR < sR+threshold )) &&
       (( *tG > sG-threshold ) && ( *tG < sG+threshold )) &&
       (( *tB > sB-threshold ) && ( *tB < sB+threshold ))
      )
      {
        *tR = R; *tG = G; *tB = B;

        floodFill(target,width,height, pX+1 , pY ,   threshold, sR , sG , sB , R , G , B ,depth+1);
        floodFill(target,width,height, pX-1 , pY ,   threshold, sR , sG , sB , R , G , B ,depth+1);

        floodFill(target,width,height, pX , pY+1 ,   threshold, sR , sG , sB , R , G , B ,depth+1);
        floodFill(target,width,height, pX , pY-1 ,   threshold, sR , sG , sB , R , G , B ,depth+1);

        floodFill(target,width,height, pX+1 , pY+1 , threshold, sR , sG , sB , R , G , B ,depth+1);
        floodFill(target,width,height, pX-1 , pY-1 , threshold, sR , sG , sB , R , G , B ,depth+1);

        floodFill(target,width,height, pX-1 , pY+1 , threshold, sR , sG , sB , R , G , B ,depth+1);
        floodFill(target,width,height, pX+1 , pY-1 , threshold, sR , sG , sB , R , G , B ,depth+1);
      }

   return 1;
}






int floodFillUShort(unsigned short * target , unsigned int width , unsigned int height ,
                    signed int pX , signed int pY , int threshold,
                    unsigned short sourceDepth ,
                    unsigned short replaceDepth , int depth)
{
 if ( (pX<0) || (pY<0) || (pX>=width) || (pY>=height) ) { return 0; }
 if (depth>2000) { return 0; }

 if (target==0) { return 0; }
 if (width==0) { return 0; }
 if (height==0) { return 0; }

 unsigned short * source = (unsigned short *) target  + ( pX + pY * width );

 unsigned short * tDepth = source; ++source;


  if (  *tDepth == replaceDepth    ) { return 0; }


  if (
       (( *tDepth > sourceDepth-threshold ) && ( *tDepth < sourceDepth+threshold ))
     )
      {
        *tDepth = replaceDepth ;

        floodFillUShort(target,width,height, pX+1 , pY ,   threshold, sourceDepth ,replaceDepth ,depth+1);
        floodFillUShort(target,width,height, pX-1 , pY ,   threshold, sourceDepth ,replaceDepth ,depth+1);

        floodFillUShort(target,width,height, pX , pY+1 ,   threshold, sourceDepth ,replaceDepth ,depth+1);
        floodFillUShort(target,width,height, pX , pY-1 ,   threshold, sourceDepth ,replaceDepth ,depth+1);

        floodFillUShort(target,width,height, pX+1 , pY+1 , threshold, sourceDepth ,replaceDepth ,depth+1);
        floodFillUShort(target,width,height, pX-1 , pY-1 , threshold, sourceDepth ,replaceDepth ,depth+1);

        floodFillUShort(target,width,height, pX-1 , pY+1 , threshold, sourceDepth ,replaceDepth ,depth+1);
        floodFillUShort(target,width,height, pX+1 , pY-1 , threshold, sourceDepth ,replaceDepth ,depth+1);
      }

   return 1;
}














