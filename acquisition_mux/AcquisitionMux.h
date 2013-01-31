#ifndef ACQUISITIONMUX_H_INCLUDED
#define ACQUISITIONMUX_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif


struct AcquisitionMuxContext
{
   unsigned int ModuleA;
   unsigned int ModuleB;

   unsigned int DeviceA;
   unsigned int DeviceB;


};


typedef unsigned int AcquisitionMultiplexerIdentifier;


int mux2RGBAndDepthFrames( char * rgbBase, char * rgbOverlay , char * rgbOut , short * depthBase, short * depthOverlay , short * depthOut , unsigned int width , unsigned int height , unsigned int mux_type);
int saveMuxImageToFile(char * filename,char * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel);

#ifdef __cplusplus
}
#endif


#endif // ACQUISITIONMUX_H_INCLUDED
