#ifndef ACQUISITIONFILEOUTPUT_H_INCLUDED
#define ACQUISITIONFILEOUTPUT_H_INCLUDED

#include "Acquisition.h"


int _acfo_savePCD_PointCloud(char * filename ,unsigned short * depthFrame ,unsigned char * colorFrame , unsigned int width , unsigned int height , float cx , float cy , float fx , float fy );



int _acfo_savePCD_PointCloudNoEmpty(char * filename ,unsigned short * depthFrame ,unsigned char * colorFrame , unsigned int width , unsigned int height , float cx , float cy , float fx , float fy );


int _acfo_acquisitionSavePCDPointCoud(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename);


int _acfo_swapEndiannessPNM(void * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel);


int _acfo_acquisitionSaveRawImageToFile(char * filename,unsigned char * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel);


int _acfo_acquisitionSaveLocationStamp(char * filename);

int _acfo_acquisitionSaveTimestamp(ModuleIdentifier moduleID,DeviceIdentifier devID,const char * filename);


unsigned char * _acfo_convertShortDepthToRGBDepth(unsigned short * depth,unsigned int width , unsigned int height);


unsigned char * _acfo_convertShortDepthToCharDepth(unsigned short * depth,unsigned int width , unsigned int height , unsigned int min_depth , unsigned int max_depth);


unsigned char * _acfo_convertShortDepthTo3CharDepth(unsigned short * depth,unsigned int width , unsigned int height , unsigned int min_depth , unsigned int max_depth);


int _acfo_acquisitionSaveColorFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename, int compress);

int _acfo_acquisitionSaveDepthFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename, int compress);


int _acfo_acquisitionSaveColoredDepthFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename);

int _acfo_acquisitionSaveDepthFrame1C(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename);

#endif // ACQUISITIONFILEOUTPUT_H_INCLUDED
