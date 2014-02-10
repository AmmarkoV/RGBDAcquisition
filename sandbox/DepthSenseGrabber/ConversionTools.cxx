#ifdef _MSC_VER
#include <windows.h>
#endif

#include <stdio.h>
#include <time.h>

#include <vector>
#include <exception>

#include "cv.h"
#include "highgui.h"

#include <DepthSense.hxx>

using namespace DepthSense;
using namespace std;


void uvToColorPixelInd(UV uv, int widthColor, int heightColor, int* colorPixelInd, int* colorPixelRow, int* colorPixelCol) {
    if(uv.u >= 0.0 && uv.u <= 1.0 && uv.v >= 0.0 && uv.v <= 1.0) {
        //int row, col;
        //int colorPixelInd;
        *colorPixelRow = (int) (uv.v * ((float) heightColor));
        *colorPixelCol = (int) (uv.u * ((float) widthColor));
        *colorPixelInd = (*colorPixelRow)*widthColor + (*colorPixelCol);
        //printf("colorPixelInd %d\n",colorPixelInd);
        //int[3] colorPixelCoord = {colorPixelInd,row,col};
        //return colorPixelCoord;
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


