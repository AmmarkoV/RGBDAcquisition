#include "projection.h"
#include <stdio.h>
#include <stdlib.h>


#include "math.h"




void outImgToXYZ(float i,float j,unsigned int face,unsigned int edge , float * x , float * y , float * z)
{
 float a = (float) 2.0*i/edge;
 float b = (float) 2.0*j/edge;

  switch(face)
  {
    case 0 : // back
      *x = -1.0;  *y=1.0-a;  *z=3.0-b;
    break;
    case 1 : // left
      *x = a-3.0;  *y=-1.0;  *z=3.0-b;
    break;
    case 2 : // front
      *x = 1.0;  *y=a - 5.0;  *z=3.0-b;
    break;
    case 3 : // right
      *x = 7.0-a;  *y=1.0;  *z=3.0-b;
    break;
    case 4 : // top
      *x = b-1.0;  *y=a -5.0;  *z=1.0;
    break;
    case 5 : // bottom
      *x = 5.0-b;  *y=a-5.0;  *z=-1.0;
    break;
  }


}



float clip(float val , float minVal , float maxVal)
{
  if ( val < minVal ) { return minVal; }
  if ( val > maxVal ) { return maxVal; }
  return val;
}

float hypotMine(x,y)
{
  return sqrt(x*x + y*y);
}




void createCubeMapFaceP(
                             char * out , unsigned int outWidth ,unsigned int outHeight , unsigned int outChannels , unsigned int outBitsPerPixel ,
                             char * in , unsigned int inWidth , unsigned int inHeight , unsigned int inChannels , unsigned int inBitsPerPixel
                            )
{
   float pi = M_PI;
   float x,y,z;

   float A[3],B[3],C[3],D[3];

   //inSize = imgIn.size
   //outSize = imgOut.size
   float edge = (float) inWidth/4;//   # the length of each edge in pixels
   unsigned int face=0;
   unsigned int face2=0;
   unsigned int i=0,j=0;
   for (i=0; i<outWidth; i++)
   {
     unsigned int face = (unsigned int)i/edge; //# 0 - back, 1 - left 2 - front, 3 - right

        unsigned int rngStart = 0;
        unsigned int rngEnd = 0;

        if (face==2) { rngStart=0;    rngEnd=edge*3; } else
                     { rngStart=edge; rngEnd=edge*2; }

        for (j=rngStart; j<rngEnd; j++)
        {
            if (j<edge)       { face2 = 4; }  else //# top
            if ( j>=2*edge )  { face2 = 5; }  else //# bottom
                              { face2 = face; }

            outImgToXYZ(i,j,face2,edge,&x,&y,&z);
            float theta = atan2(y,x); // # range -pi to pi
            float r = hypot(x,y);
            float phi = atan2(z,r); // # range -pi/2 to pi/2
            // source img coords
            float thetaPi = (float) (theta + pi)/pi;
            float uf = (float) ( 2.0*edge* thetaPi);

            float piphiDivPi = (float) (pi/2 - phi)/pi;
            float vf = (float) ( 2.0*edge * piphiDivPi);
            // Use bilinear interpolation between the four surrounding pixels
            float ui = floor(uf);//  # coord of pixel to bottom left
            float vi = floor(vf);
            float u2 = ui+1.0;//       # coords of pixel to top right
            float v2 = vi+1.0;
            float mu = (float) uf-ui;//     # fraction of way across pixel
            float nu = (float) vf-vi;
            // Pixel values of four corners
            unsigned int xA = (unsigned int) ui % inWidth, yA = (unsigned int ) clip(vi,0,inHeight-1);
            unsigned int memplaceA =yA * inWidth*3 + xA*3;
            if (memplaceA>=inWidth*inHeight*3) { memplaceA=0; }
            A[0] = in[memplaceA+0];
            A[1] = in[memplaceA+1];
            A[2] = in[memplaceA+2];

            unsigned int xB = (unsigned int) u2 % inWidth, yB = (unsigned int ) clip(vi,0,inHeight-1);
            unsigned int memplaceB =yB * inWidth *3 + xB *3;
            if (memplaceB>=inWidth*inHeight*3) { memplaceB=0; }
            B[0] = in[memplaceB+0];
            B[1] = in[memplaceB+1];
            B[2] = in[memplaceB+2];

            unsigned int xC = (unsigned int) ui % inWidth, yC = (unsigned int ) clip(v2,0,inHeight-1);
            unsigned int memplaceC =yC * inWidth *3 + xC *3;
            if (memplaceC>=inWidth*inHeight*3) { memplaceC=0; }
            C[0] = in[memplaceC+0];
            C[1] = in[memplaceC+1];
            C[2] = in[memplaceC+2];

            unsigned int xD = (unsigned int) u2 % inWidth, yD = (unsigned int ) clip(v2,0,inHeight-1);
            unsigned int memplaceD =yD * inWidth *3 + xD *3;
            if (memplaceD>=inWidth*inHeight*3) { memplaceD=0; }
            D[0] = in[memplaceD+0];
            D[1] = in[memplaceD+1];
            D[2] = in[memplaceD+2];
            // interpolate

        float rCol= A[0]*(1-mu)*(1-nu) + B[0]*(mu)*(1-nu) + C[0]*(1-mu)*nu + D[0]*mu*nu;
        if (rCol>255) { rCol=255; }

        float gCol= A[1]*(1-mu)*(1-nu) + B[1]*(mu)*(1-nu) + C[1]*(1-mu)*nu + D[1]*mu*nu;
        if (gCol>255) { gCol=255; }

        float bCol= A[2]*(1-mu)*(1-nu) + B[2]*(mu)*(1-nu) + C[2]*(1-mu)*nu + D[2]*mu*nu;
        if (bCol>255) { bCol=255; }

       //outPix[i,j] = (int(round(r)),int(round(g)),int(round(b)))
       unsigned int memplaceOUT =j * outWidth *3 + i *3;
       if (memplaceOUT>=outWidth*outHeight*3) { memplaceOUT=0; }
       out[memplaceOUT+ 0 ] = rCol;
       out[memplaceOUT+ 1 ] = gCol;
       out[memplaceOUT + 2 ] = bCol;


       }
   }
}



void createCubeMapFace(char * out , unsigned int outWidth ,unsigned int outHeight , unsigned int outChannels , unsigned int outBitsPerPixel ,
                       char * in , unsigned int inWidth , unsigned int inHeight , unsigned int inChannels , unsigned int inBitsPerPixel
                        )
{
   createCubeMapFaceP( out , outWidth , outHeight , outChannels ,  outBitsPerPixel ,
                        in , inWidth , inHeight , inChannels , inBitsPerPixel
                        );

}

void getCubeMap2DCoords(unsigned int inputWidth , unsigned int inputHeight , float x ,float y , float z , unsigned int * outX ,unsigned int * outY , unsigned int *outWidth , unsigned int * outHeight )
{
   *outWidth = (unsigned int ) inputWidth / 4;
   *outHeight = (unsigned int ) inputHeight / 3;

   if ( (x<0) && (y==0) && (z==0) ) { *outX=*outWidth*1;    *outY=*outHeight; }  else
   if ( (x>0) && (y==0) && (z==0) ) { *outX=*outWidth*3;    *outY=*outHeight; }  else

   if ( (x==0) && (y<0) && (z==0) ) { *outX=*outWidth*2 ;    *outY=*outHeight*2; }  else
   if ( (x==0) && (y>0) && (z==0) ) { *outX=*outWidth*2 ;    *outY=*outHeight*0; }  else

   if ( (x==0) && (y==0) && (z<0) ) { *outX=*outWidth*0;    *outY=*outHeight*1; }  else
   if ( (x==0) && (y==0) && (z>0) ) { *outX=*outWidth*2;    *outY=*outHeight*1; }


}
