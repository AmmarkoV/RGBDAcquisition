#include "projection.h"
#include <stdio.h>
#include <stdlib.h>


#include "math.h"
// Define our six cube faces.
// 0 - 3 are side faces, clockwise order
// 4 and 5 are top and bottom, respectively
float faceTransform[6][2] =
{
    {0, 0},
    {M_PI / 2, 0},
    {M_PI, 0},
    {-M_PI / 2, 0},
    {0, -M_PI / 2},
    {0, M_PI / 2}
};

// Map a part of the equirectangular panorama (in) to a cube face
// (face). The ID of the face is given by faceId. The desired
// width and height are given by width and height.


void createCubeMapFaceCV(char * out , unsigned int outWidth ,unsigned int outHeight , unsigned int outChannels , unsigned int outBitsPerPixel ,
                       char * in , unsigned int inWidth , unsigned int inHeight , unsigned int inChannels , unsigned int inBitsPerPixel
                        )
{
 // http://stackoverflow.com/questions/29678510/convert-21-equirectangular-panorama-to-cube-map
 unsigned int x=0, y=0;
 unsigned int faceId = 0;

 unsigned int fLoop=0;

 unsigned int width = outWidth;
 unsigned int height = outHeight;

    float * mapx = (float * ) malloc(outWidth * outHeight * sizeof(float));
    if (mapx==0) { return ; }
    float * mapy = (float * ) malloc(outWidth * outHeight * sizeof(float));
    if (mapy==0) { free(mapx); return ; }

  for (fLoop=0; fLoop<6; fLoop++)
  {
    faceId=fLoop;
    // Allocate map



    // Calculate adjacent (ak) and opposite (an) of the
    // triangle that is spanned from the sphere center
    //to our cube face.
    const float an = sin(M_PI / 4);
    const float ak = cos(M_PI / 4);

    const float ftu = faceTransform[faceId][0];
    const float ftv = faceTransform[faceId][1];

    // For each point in the target image,
    // calculate the corresponding source coordinates.

    for(y = 0; y < height; y++)
     {
      for(x = 0; x < width; x++)
      {
        // Map face pixel coordinates to [-1, 1] on plane
        float nx =  (float)y / (float)height - 0.5f;
        float ny =  (float)x / (float)width  - 0.5f;

//        nx *= 2;
 //       ny *= 2;

        // Map [-1, 1] plane coords to [-an, an]
        // thats the coordinates in respect to a unit sphere
        // that contains our box.
        nx *= 2*an;
        ny *= 2*an;

        float u =0 , v=0;

        // Project from plane to sphere surface.
        if(ftv == 0)
            {  // Center faces
                u = atan2(nx, ak);
                v = atan2(ny * cos(u), ak);
                u += ftu;
            } else
        if(ftv > 0)
            {   // Bottom face
                float d = sqrt(nx * nx + ny * ny);
                v = M_PI / 2 - atan2(d, ak);
                u = atan2(ny, nx);
            } else
            {   // Top face
                float d = sqrt(nx * nx + ny * ny);
                v = -M_PI / 2 + atan2(d, ak);
                u = atan2(-ny, nx);
            }

            // Map from angular coordinates to [-1, 1], respectively.
            u = u / (M_PI);
            v = v / (M_PI / 2);

            // Warp around, if our coordinates are out of bounds.
            while (v < -1)  { v += 2; u += 1; }
            while (v > 1)   { v -= 2; u += 1; }

            while(u < -1) { u += 2; }
            while(u > 1)  { u -= 2; }

            // Map from [-1, 1] to in texture space
            u = (float) (u / 2.0f) + 0.5f;     v = (float) (v / 2.0f) + 0.5f;
            u = u * (float) (inWidth - 1);   v = v * (float) (inHeight - 1);

            // Save the result for this pixel in map
            mapx[y*outWidth + x] = u;
            mapy[y*outWidth + x] = v;

            //mapx.at<float>(x, y) = u;
            //mapy.at<float>(x, y) = v;
        }
    }

    // Recreate output image if it has wrong size or type.
    //if(face.cols != width || face.rows != height ||
    //    face.type() != in.type()) {
    //    face = Mat(width, height, in.type());
    //}
fprintf(stderr,"Reconstructing projections face %u \n",faceId);
for(y = 0; y < height; y++)
     {
      for(x = 0; x < width; x++)
      {
          unsigned int outX =  (unsigned int) mapx[y*outWidth + x];
          unsigned int outY =  (unsigned int) mapy[y*outWidth + x];

          if (outX>=outWidth)  { outX=outWidth-1;  }
          if (outY>=outHeight) { outY=outHeight-1; }
          unsigned int inPlace = y * inWidth + x;
          unsigned int outPlace = outY * outWidth + outX;
          out[outPlace]=in[inPlace]; ++outPlace; // ++inPlace;
          out[outPlace]=in[inPlace]; ++outPlace; //++inPlace;
          out[outPlace]=in[inPlace];
      }
     }


  }


    free(mapx);
    free(mapy);

    // Do actual resampling using OpenCV's remap
    // remap(in, face, mapx, mapy, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
   /* */
}





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




void createCubeMapFace(char * out , unsigned int outWidth ,unsigned int outHeight , unsigned int outChannels , unsigned int outBitsPerPixel ,
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
            unsigned int memplaceA =yA * inWidth + xA;
            A[0] = in[memplaceA+0];
            A[1] = in[memplaceA+1];
            A[2] = in[memplaceA+2];

            unsigned int xB = (unsigned int) u2 % inWidth, yB = (unsigned int ) clip(vi,0,inHeight-1);
            unsigned int memplaceB =yB * inWidth + xB;
            B[0] = in[memplaceB+0];
            B[1] = in[memplaceB+1];
            B[2] = in[memplaceB+2];

            unsigned int xC = (unsigned int) ui % inWidth, yC = (unsigned int ) clip(v2,0,inHeight-1);
            unsigned int memplaceC =yC * inWidth + xC;
            C[0] = in[memplaceC+0];
            C[1] = in[memplaceC+1];
            C[2] = in[memplaceC+2];

            unsigned int xD = (unsigned int) u2 % inWidth, yD = (unsigned int ) clip(v2,0,inHeight-1);
            unsigned int memplaceD =yD * inWidth + xD;
            D[0] = in[memplaceD+0];
            D[1] = in[memplaceD+1];
            D[2] = in[memplaceD+2];
            // interpolate

        float rCol= A[0]*(1-mu)*(1-nu) + B[0]*(mu)*(1-nu) + C[0]*(1-mu)*nu + D[0]*mu*nu;
        float gCol= A[1]*(1-mu)*(1-nu) + B[1]*(mu)*(1-nu) + C[1]*(1-mu)*nu + D[1]*mu*nu;
        float bCol= A[2]*(1-mu)*(1-nu) + B[2]*(mu)*(1-nu) + C[2]*(1-mu)*nu + D[2]*mu*nu;

       //outPix[i,j] = (int(round(r)),int(round(g)),int(round(b)))
       out[j*outWidth + i + 0 ] = rCol;
       out[j*outWidth + i + 1 ] = gCol;
       out[j*outWidth + i + 2 ] = bCol;


       }
   }
}


