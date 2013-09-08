#ifndef COMBINERGBANDDEPTHOUTPUT_H_INCLUDED
#define COMBINERGBANDDEPTHOUTPUT_H_INCLUDED


enum combinationModesEnumerator
{
  COMBINE_AND=0 ,
  COMBINE_OR    ,
  COMBINE_XOR   ,
  COMBINE_KEEP_ONLY_RGB  ,
  COMBINE_KEEP_ONLY_DEPTH ,
  //-----------------------------
  NUMBER_OF_COMBINATION_MODES
};


unsigned char * combineRGBAndDepthToOutput( unsigned char * selectedRGB , unsigned char * selectedDepth , int combinationMode, unsigned int width , unsigned int height );

#endif // COMBINERGBANDDEPTHOUTPUT_H_INCLUDED
