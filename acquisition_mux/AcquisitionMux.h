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


int mux2RGBAndDepthFrames( char * rgb1, char * rgb2 , char * rgbOut , short * depth1, short * depth2 , short * depthOut , unsigned int width , unsigned int height , unsigned int mux_type);


#ifdef __cplusplus
}
#endif


#endif // ACQUISITIONMUX_H_INCLUDED
