#ifdef _MSC_VER
#include <windows.h>
#endif

#include <stdio.h>
#include <time.h>

#include <vector>
#include <exception>

#include <DepthSense.hxx>

using namespace DepthSense;
using namespace std;


void uvToColorPixelInd(UV uv, int widthColor, int heightColor, int* colorPixelInd, int* colorPixelRow, int* colorPixelCol) {
    if(uv.u > 0.00001 && uv.u < 0.99999 && uv.v > 0.00001 && uv.v < 0.99999) {
        *colorPixelRow = (int) (uv.v * ((float) heightColor) + 0.5);
        *colorPixelCol = (int) (uv.u * ((float) widthColor) + 0.5);
        /*float row = uv.v * ((float) heightColor);
        float col = uv.u * ((float) widthColor);
        float ind = (row * ((float) widthColor)) + col;
        *colorPixelInd = (int) ind;*/
        *colorPixelInd = (*colorPixelRow)*widthColor + *colorPixelCol;
        //printf("TEST\n %f \n %i \n %i\n",ind,(int) ind, *colorPixelInd);
        //*colorPixelInd = (int) (((uv.v * ((float) heightColor))*((float) widthColor)) + (uv.u * ((float) widthColor)));
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

void saveRawColorFrame(char* fileName, uint8_t* pixels, int width, int height, int timeStamp)
{
    FILE *pFile=0;
    pFile = fopen(fileName,"wb");

    if (pFile!=0)
    {
        fprintf(pFile, "P6\n");
        /*
        char timeStampStr[256]={0};
        GetDateString(timeStampStr,"TIMESTAMP",1,0,0,0,0,0,0,0);
        fprintf(fd, "#%s\n", timeStampStr );*/

        //fprintf(fd, "#TIMESTAMP %lu\n",GetTickCount());
        fprintf(pFile, "#TIMESTAMP %i\n",timeStamp);
        fprintf(pFile, "%d %d\n%i\n", width, height, 255);
        fwrite(pixels,1,3*width*height,pFile);
        fflush(pFile);
        fclose(pFile);
    }
}



void saveRawDepthFrame(char* fileName, uint16_t* pixels, int width, int height, int timeStamp)
{
    FILE *pFile=0;
    pFile = fopen(fileName,"wb");

    if (pFile!=0)
    {
        fprintf(pFile, "P5\n");
        /*
        char timeStampStr[256]={0};
        GetDateString(timeStampStr,"TIMESTAMP",1,0,0,0,0,0,0,0);
        fprintf(fd, "#%s\n", timeStampStr );*/

        //fprintf(fd, "#TIMESTAMP %lu\n",GetTickCount());
        fprintf(pFile, "#TIMESTAMP %i\n",timeStamp);

        fprintf(pFile, "%d %d\n%i\n", width, height, 65535);
        fwrite(pixels,2,width*height,pFile);
        fflush(pFile);
        fclose(pFile);
    }
}

void doubleSizeDepth(uint16_t* src, uint16_t* dst, int srcWidth, int srcHeight) {
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

void doubleSizeUV(UV* src, UV* dst, int srcWidth, int srcHeight) {
    for (int i = 0; i < srcHeight; i++)
        for (int j = 0; j < srcWidth; j++) {
            dst[(2*i)*2*srcWidth+2*j] = UV(src[i*srcWidth+j].u,src[i*srcWidth+j].v);
        }
    for (int i = 0; i < srcHeight-1; i++)
        for (int j = 0; j < srcWidth-1; j++) {
            dst[(2*i+1)*2*srcWidth+2*j] = UV(0.5*(src[i*srcWidth+j].u+src[(i+1)*srcWidth+j].u),0.5*(src[i*srcWidth+j].v+src[(i+1)*srcWidth+j].v));
            dst[(2*i)*2*srcWidth+2*j+1] = UV(0.5*(src[i*srcWidth+j].u+src[(i)*srcWidth+j+1].u),0.5*(src[i*srcWidth+j].v+src[(i)*srcWidth+j+1].v));
        }
    for (int i = 0; i < srcHeight-1; i++)
        for (int j = 0; j < srcWidth-1; j++) {
            dst[(2*i+1)*2*srcWidth+2*j+1] = UV(0.25*(src[i*srcWidth+j].u+src[(i+1)*srcWidth+j].u+src[(i)*srcWidth+j+1].u+src[(i+1)*srcWidth+j+1].u),0.25*(src[i*srcWidth+j].v+src[(i+1)*srcWidth+j].v+src[(i)*srcWidth+j+1].v+src[(i+1)*srcWidth+j+1].v));
        }
}
