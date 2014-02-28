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

void calcDepthToPosMat(float* depthToPosMatX, float* depthToPosMatY, int fovHorizontalDeg, int fovVerticalDeg, int width, int height);
float packRGB(uint8_t* rgb);
int packRGBA(uint8_t* rgb);

void uvToColorPixelInd(UV uv, int widthColor, int heightColor, int* colorPixelInd, int* colorPixelRow, int* colorPixelCol);
FrameFormat formatName(int resType);
int formatResX(int resType);
int formatResY(int resType);

void saveRawColorFrame(char* fileName, uint8_t* pixels, int width, int height, int timeStamp);
void saveRawDepthFrame(char* fileName, uint16_t* pixels, int width, int height, int timeStamp);

void doubleSizeDepth(uint16_t* src, uint16_t* dst, int srcWidth, int srcHeight);
void doubleSizeUV(UV* src, UV* dst, int srcWidth, int srcHeight);

void rescaleMap(uint16_t* src, uint16_t* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight);
void rescaleMap(const short int* src, uint16_t* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight);
void rescaleMap(float* src, float* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight);
void rescaleMap(UV* src, UV* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight);
void rescaleMap(DepthSense::Pointer<DepthSense::UV> src, UV* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight);
void rescaleMap(FPVertex* src, FPVertex* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight);
void rescaleMap(const short int* src, uint16_t* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight, uint16_t* confidenceMap, uint16_t confidenceMin, uint16_t noDepthValue);
