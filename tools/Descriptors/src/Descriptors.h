#ifndef DESCRIPTORS_H_INCLUDED
#define DESCRIPTORS_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif


enum descriptorType
{
  EMPTY=0,
  CIRCLE ,
  LINE   ,
  PLANE  ,
  //---------
  NUMBER_OF_DESCRIPTOR_TYPES
};



struct descriptorPoint
{
  float x,y,z;
  unsigned int descriptorType;
  float dimX,dimY,dimZ;
  float rotX,rotY,rotZ;

};


struct descriptorContext
{
   unsigned int currentNumberOfPoints;
   unsigned int maximumNumberOfPoints;

   struct descriptorPoint point[640*480];


   unsigned int width,height;

};


struct descriptorContext * descriptorCreate(unsigned char * rgb , unsigned int rgbWidth ,unsigned int rgbHeight ,
                                             unsigned short * depth  , unsigned int depthWidth , unsigned int depthHeight );


unsigned char * descriptorVisualizeRGB(struct descriptorContext * desc, unsigned int * width, unsigned int * height);



#ifdef __cplusplus
}
#endif


#endif // DESCRIPTORS_H_INCLUDED
