#ifdef _MSC_VER
#include <windows.h>
#endif

#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>

#include <vector>
#include <exception>

#include <DepthSense.hxx>

using namespace DepthSense;
using namespace std;

void calcDepthToPosMat(float* depthToPosMatX, float* depthToPosMatY, int fovHorizontalDeg, int fovVerticalDeg, int width, int height) {
    double halfFovHorizontalRad = fovHorizontalDeg*M_PI/360.0;
    double halfFovVerticalRad = fovVerticalDeg*M_PI/360.0;
    double stepHorizontal = 2.0*tan(halfFovHorizontalRad)/((double) width);
    double stepVertical = 2.0*tan(halfFovVerticalRad)/((double) height);
    double startHorizontal = -tan(halfFovHorizontalRad);
    double startVertical = tan(halfFovVerticalRad);
    int currentPixelInd = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            depthToPosMatX[currentPixelInd] = (float) (startHorizontal + ((float) j)*stepHorizontal);
            depthToPosMatY[currentPixelInd] = (float) (startVertical - ((float) i)*stepVertical);
            currentPixelInd++;
        }
    }
}

float packRGB(uint8_t* rgb) {
    uint32_t rgbInt = ((uint32_t)rgb[0] << 16 | (uint32_t)rgb[1] << 8 | (uint32_t)rgb[2]);
    return *reinterpret_cast<float*>(&rgbInt);
}

int packRGBA(uint8_t* rgb) {
    int rgbInt = ((uint32_t)rgb[0] << 16 | (uint32_t)rgb[1] << 8 | (uint32_t)rgb[2]);
    return rgbInt;
}

void uvToColorPixelInd(UV uv, int widthColor, int heightColor, int* colorPixelInd, int* colorPixelRow, int* colorPixelCol) {
    if(uv.u > 0.001 && uv.u < 0.999 && uv.v > 0.001 && uv.v < 0.999) {
        *colorPixelRow = (int) (uv.v * ((float) heightColor));
        *colorPixelCol = (int) (uv.u * ((float) widthColor));
        *colorPixelInd = (*colorPixelRow)*widthColor + *colorPixelCol;
    }
    else
        *colorPixelInd = -1;
}


FrameFormat formatName(int resType) {
    switch (resType) {
        case 0: return FRAME_FORMAT_QQVGA;
        case 1: return FRAME_FORMAT_QVGA;
        case 2: return FRAME_FORMAT_VGA;
        case 3: return FRAME_FORMAT_WXGA_H;
        case 4: return FRAME_FORMAT_NHD;
        default: printf("Unsupported resolution parameter\n"); return FRAME_FORMAT_WXGA_H;
    }
}
int formatResX(int resType) {
    switch (resType) {
        case 0: return 160;
        case 1: return 320;
        case 2: return 640;
        case 3: return 1280;
        case 4: return 640;
        default: printf("Unsupported resolution parameter\n"); return 0;
    }
}
int formatResY(int resType) {
    switch (resType) {
        case 0: return 120;
        case 1: return 240;
        case 2: return 480;
        case 3: return 720;
        case 4: return 360;
        default: printf("Unsupported resolution parameter\n"); return 0;
    }
}

void saveColorFramePNM(char* fileName, uint8_t* pixels, int width, int height, int timeStamp)
{
    FILE *pFile=0;
    pFile = fopen(fileName,"wb");

    if (pFile!=0)
    {
        fprintf(pFile, "P6\n");
        fprintf(pFile, "#TIMESTAMP %i\n",timeStamp);
        fprintf(pFile, "%d %d\n%i\n", width, height, 255);
        fwrite(pixels,1,3*width*height,pFile);
        fflush(pFile);
        fclose(pFile);
    }
}





int swapDepthEndianness( uint16_t* pixels, int width, int height)
{
  if (pixels==0) { return 0; }

  unsigned char * traverser=(unsigned char * ) pixels;
  unsigned char * traverserSwap1=(unsigned char * ) pixels;
  unsigned char * traverserSwap2=(unsigned char * ) pixels;

  unsigned int bytesperpixel = 2;
  unsigned char * endOfMem = traverser + width * height * 1 * bytesperpixel;

  unsigned char tmp ;
  while ( ( traverser < endOfMem)  )
  {
    traverserSwap1 = traverser;
    traverserSwap2 = traverser+1;

    tmp = *traverserSwap1;
    *traverserSwap1 = *traverserSwap2;
    *traverserSwap2 = tmp;

    traverser += bytesperpixel;
  }

 return 1;
}


void saveDepthFramePNM(char* fileName, uint16_t* pixels, int width, int height, int timeStamp)
{
    FILE *pFile=0;
    pFile = fopen(fileName,"wb");

    if (pFile!=0)
    {
        fprintf(pFile, "P5\n");
        fprintf(pFile, "#TIMESTAMP %i\n",timeStamp);
        fprintf(pFile, "%d %d\n%i\n", width, height, 65535);

         //We have to swap the byte order to save "correctly Depth PNM" , this of course is slower
         //but it is the "right thing to do"
         swapDepthEndianness( pixels, width, height);

        fwrite(pixels,2,width*height,pFile);
        fflush(pFile);
        fclose(pFile);
    }
}

void floatSizeDepth(uint16_t* src, uint16_t* dst, int srcWidth, int srcHeight) {
    for (int i = 0; i < srcHeight; i++)
        for (int j = 0; j < srcWidth; j++) {
            dst[(2*i)*2*srcWidth+2*j] = src[i*srcWidth+j];
        }
    for (int i = 0; i < srcHeight-1; i++)
        for (int j = 0; j < srcWidth-1; j++) {
            dst[(2*i+1)*2*srcWidth+2*j] = (src[i*srcWidth+j]+src[(i+1)*srcWidth+j])/2;
            dst[(2*i)*2*srcWidth+2*j+1] = (src[i*srcWidth+j]+src[(i)*srcWidth+j+1])/2;
        }
    for (int i = 0; i < srcHeight-1; i++)
        for (int j = 0; j < srcWidth-1; j++) {
            dst[(2*i+1)*2*srcWidth+2*j+1] = (src[i*srcWidth+j]+src[(i+1)*srcWidth+j]+src[(i)*srcWidth+j+1]+src[(i+1)*srcWidth+j+1])/4;
        }
}

void floatSizeUV(UV* src, UV* dst, int srcWidth, int srcHeight) {
    for (int i = 0; i < srcHeight; i++)
        for (int j = 0; j < srcWidth; j++) {
            dst[(2*i)*2*srcWidth+2*j] = UV(src[i*srcWidth+j].u,src[i*srcWidth+j].v);
        }
    for (int i = 0; i < srcHeight-1; i++)
		for (int j = 0; j < srcWidth - 1; j++) {
			dst[(2 * i + 1) * 2 * srcWidth + 2 * j] = UV(0.5f*(src[i*srcWidth + j].u + src[(i + 1)*srcWidth + j].u), 0.5f*(src[i*srcWidth + j].v + src[(i + 1)*srcWidth + j].v));
            dst[(2*i)*2*srcWidth+2*j+1] = UV(0.5f*(src[i*srcWidth+j].u+src[(i)*srcWidth+j+1].u),0.5f*(src[i*srcWidth+j].v+src[(i)*srcWidth+j+1].v));
        }
    for (int i = 0; i < srcHeight-1; i++)
        for (int j = 0; j < srcWidth-1; j++) {
            dst[(2*i+1)*2*srcWidth+2*j+1] = UV(0.25f*(src[i*srcWidth+j].u+src[(i+1)*srcWidth+j].u+src[(i)*srcWidth+j+1].u+src[(i+1)*srcWidth+j+1].u),
				                               0.25f*(src[i*srcWidth+j].v+src[(i+1)*srcWidth+j].v+src[(i)*srcWidth+j+1].v+src[(i+1)*srcWidth+j+1].v));
        }
}

void rescaleMap(float* src, float* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {

    float stepWidth=(float)(srcWidth-1)/(float)(dstWidth-1);
    float stepHeight=(float)(srcHeight-1)/(float)(dstHeight-1);

    for (int x=0;x<dstWidth;x++)
    {
        float fx=x*stepWidth;
        float dx=fx-(int)fx;
        int ffx = (int) floor(fx);
		int cfx = (int)ceil(fx);
        for (int y=0;y<dstHeight;y++)
        {
            float fy=y*stepHeight;
            float dy=fy-(int)fy;
			int ffy = (int)floor(fy);
			int cfy = (int) ceil(fy);

            float val1, val2, val3, val4;
            val1 = src[ffx + ffy*srcWidth];
            val2 = src[cfx + ffy*srcWidth];
            val3 = src[ffx + cfy*srcWidth];
            val4 = src[cfx + cfy*srcWidth];

            float valT1 = dx*val2 + (1-dx)*val1;
            float valT2 = dx*val4 + (1-dx)*val3;

            float val = dy*valT2 + (1-dy)*valT1;

            dst[x + y*dstWidth] = val;
        }
    }
}


void rescaleMap(FPVertex* src, FPVertex* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {

    float stepWidth=(float)(srcWidth-1)/(float)(dstWidth-1);
    float stepHeight=(float)(srcHeight-1)/(float)(dstHeight-1);

    for (int x=0;x<dstWidth;x++)
    {
        float fx=x*stepWidth;
        float dx=fx-(int)fx;
		int ffx = (int) floor(fx);
		int cfx = (int) ceil(fx);
        for (int y=0;y<dstHeight;y++)
        {
            float fy=y*stepHeight;
            float dy=fy-(int)fy;
			int ffy = (int)floor(fy);
			int cfy = (int) ceil(fy);

            float val1x, val2x, val3x, val4x;
            val1x = src[ffx + ffy*srcWidth].x;
            val2x = src[cfx + ffy*srcWidth].x;
            val3x = src[ffx + cfy*srcWidth].x;
            val4x = src[cfx + cfy*srcWidth].x;
            float valT1x = dx*val2x + (1-dx)*val1x;
            float valT2x = dx*val4x + (1-dx)*val3x;
            float valx = dy*valT2x + (1-dy)*valT1x;

            float val1y, val2y, val3y, val4y;
            val1y = src[ffx + ffy*srcWidth].y;
            val2y = src[cfx + ffy*srcWidth].y;
            val3y = src[ffx + cfy*srcWidth].y;
            val4y = src[cfx + cfy*srcWidth].y;
            float valT1y = dx*val2y + (1-dx)*val1y;
            float valT2y = dx*val4y + (1-dx)*val3y;
            float valy = dy*valT2y + (1-dy)*valT1y;

            float val1z, val2z, val3z, val4z;
            val1z = src[ffx + ffy*srcWidth].z;
            val2z = src[cfx + ffy*srcWidth].z;
            val3z = src[ffx + cfy*srcWidth].z;
            val4z = src[cfx + cfy*srcWidth].z;
            float valT1z = dx*val2z + (1-dx)*val1z;
            float valT2z = dx*val4z + (1-dx)*val3z;
            float valz = dy*valT2z + (1-dy)*valT1z;

            dst[x + y*dstWidth] = FPVertex(valx,valy,valz);
        }
    }
}





void rescaleMap(uint16_t* src, uint16_t* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {

    float stepWidth=(float)(srcWidth-1)/(float)(dstWidth-1);
    float stepHeight=(float)(srcHeight-1)/(float)(dstHeight-1);

    for (int x=0;x<dstWidth;x++)
    {
        float fx=x*stepWidth;
        float dx=fx-(int)fx;
        int ffx = (int) floor(fx);
		int cfx = (int) ceil(fx);
        for (int y=0;y<dstHeight;y++)
        {
            float fy=y*stepHeight;
            float dy=fy-(int)fy;
			int ffy = (int) floor(fy);
			int cfy = (int) ceil(fy);

            uint16_t val1, val2, val3, val4;
            val1 = src[ffx + ffy*srcWidth];
            val2 = src[cfx + ffy*srcWidth];
            val3 = src[ffx + cfy*srcWidth];
            val4 = src[cfx + cfy*srcWidth];

            float valT1 = dx*val2 + (1-dx)*val1;
            float valT2 = dx*val4 + (1-dx)*val3;

            uint16_t val = (uint16_t) (dy*valT2 + (1-dy)*valT1);

            dst[x + y*dstWidth] = val;
        }
    }
}



void rescaleMap(const short int* src, uint16_t* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {

    float stepWidth=(float)(srcWidth-1)/(float)(dstWidth-1);
    float stepHeight=(float)(srcHeight-1)/(float)(dstHeight-1);

    for (int x=0;x<dstWidth;x++)
    {
        float fx=x*stepWidth;
        float dx=fx-(int)fx;
		int ffx = (int) floor(fx);
		int cfx = (int) ceil(fx);
        for (int y=0;y<dstHeight;y++)
        {
            float fy=y*stepHeight;
            float dy=fy-(int)fy;
			int ffy = (int) floor(fy);
			int cfy = (int) ceil(fy);

            uint16_t val1, val2, val3, val4;
            val1 = src[ffx + ffy*srcWidth];
            val2 = src[cfx + ffy*srcWidth];
            val3 = src[ffx + cfy*srcWidth];
            val4 = src[cfx + cfy*srcWidth];

            float valT1 = dx*val2 + (1-dx)*val1;
            float valT2 = dx*val4 + (1-dx)*val3;

            uint16_t val = (uint16_t) (dy*valT2 + (1-dy)*valT1);

            dst[x + y*dstWidth] = val;
        }
    }
}


void rescaleMap(const short int* src, uint16_t* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight, uint16_t* confidenceMap, uint16_t confidenceMin, uint16_t noDepthValue) {

    float stepWidth=(float)(srcWidth-1)/(float)(dstWidth-1);
    float stepHeight=(float)(srcHeight-1)/(float)(dstHeight-1);

    for (int x=0;x<dstWidth;x++)
    {
        float fx=x*stepWidth;
        float dx=fx-(int)fx;
		int ffx = (int) floor(fx);
		int cfx = (int) ceil(fx);
        for (int y=0;y<dstHeight;y++)
        {
			if (confidenceMap[x + y*dstWidth] > confidenceMin) {
				float fy=y*stepHeight;
		        float dy=fy-(int)fy;
				int ffy = (int) floor(fy);
				int cfy = (int) ceil(fy);

		        uint16_t val1, val2, val3, val4;
		        val1 = src[ffx + ffy*srcWidth];
		        val2 = src[cfx + ffy*srcWidth];
		        val3 = src[ffx + cfy*srcWidth];
		        val4 = src[cfx + cfy*srcWidth];

		        float valT1 = dx*val2 + (1-dx)*val1;
		        float valT2 = dx*val4 + (1-dx)*val3;

				uint16_t val = (uint16_t) (dy*valT2 + (1 - dy)*valT1);

		        dst[x + y*dstWidth] = val;
			}
			else {
				dst[x + y*dstWidth] = noDepthValue;
			}
        }
    }
}



void rescaleMap(UV* src, UV* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {

    float stepWidth=(float)(srcWidth-1)/(float)(dstWidth-1);
    float stepHeight=(float)(srcHeight-1)/(float)(dstHeight-1);

    for (int x=0;x<dstWidth;x++)
    {
        float fx=x*stepWidth;
        float dx=fx-(int)fx;
		int ffx = (int) floor(fx);
		int cfx = (int) ceil(fx);
        for (int y=0;y<dstHeight;y++)
        {
            float fy=y*stepHeight;
            float dy=fy-(int)fy;
			int ffy = (int) floor(fy);
			int cfy = (int) ceil(fy);

            float u1, u2, u3, u4, v1, v2, v3, v4;
            u1 = src[ffx + ffy*srcWidth].u;
            u2 = src[cfx + ffy*srcWidth].u;
            u3 = src[ffx + cfy*srcWidth].u;
            u4 = src[cfx + cfy*srcWidth].u;
            v1 = src[ffx + ffy*srcWidth].v;
            v2 = src[cfx + ffy*srcWidth].v;
            v3 = src[ffx + cfy*srcWidth].v;
            v4 = src[cfx + cfy*srcWidth].v;

            float uT1 = dx*u2 + (1-dx)*u1;
            float uT2 = dx*u4 + (1-dx)*u3;
            float vT1 = dx*v2 + (1-dx)*v1;
            float vT2 = dx*v4 + (1-dx)*v3;

            float u = dy*uT2 + (1-dy)*uT1;
            float v = dy*vT2 + (1-dy)*vT1;

            dst[x + y*dstWidth] = UV(u,v);
        }
    }
}


void rescaleMap(DepthSense::Pointer<DepthSense::UV> src, UV* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {

    float stepWidth=(float)(srcWidth-1)/(float)(dstWidth-1);
    float stepHeight=(float)(srcHeight-1)/(float)(dstHeight-1);

    for (int x=0;x<dstWidth;x++)
    {
        float fx=x*stepWidth;
        float dx=fx-(int)fx;
		int ffx = (int) floor(fx);
		int cfx = (int) ceil(fx);
        for (int y=0;y<dstHeight;y++)
        {
            float fy=y*stepHeight;
            float dy=fy-(int)fy;
			int ffy = (int) floor(fy);
			int cfy = (int) ceil(fy);

            float u1, u2, u3, u4, v1, v2, v3, v4;
            u1 = src[ffx + ffy*srcWidth].u;
            u2 = src[cfx + ffy*srcWidth].u;
            u3 = src[ffx + cfy*srcWidth].u;
            u4 = src[cfx + cfy*srcWidth].u;
            v1 = src[ffx + ffy*srcWidth].v;
            v2 = src[cfx + ffy*srcWidth].v;
            v3 = src[ffx + cfy*srcWidth].v;
            v4 = src[cfx + cfy*srcWidth].v;

            float uT1 = dx*u2 + (1-dx)*u1;
            float uT2 = dx*u4 + (1-dx)*u3;
            float vT1 = dx*v2 + (1-dx)*v1;
            float vT2 = dx*v4 + (1-dx)*v3;

            float u = dy*uT2 + (1-dy)*uT1;
            float v = dy*vT2 + (1-dy)*vT1;

            dst[x + y*dstWidth] = UV(u,v);
        }
    }
}
