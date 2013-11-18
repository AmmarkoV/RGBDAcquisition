#ifndef OCR_H_INCLUDED
#define OCR_H_INCLUDED


#include "patternSets.h"

int doOCR(
           unsigned char * screen , unsigned int screenWidth , unsigned int screenHeight ,
           unsigned int sX , unsigned int sY  , unsigned int width , unsigned int height,

           struct PatternSet * font ,

           char * output ,
           unsigned int outputMaxLength
          );


#endif // OCR_H_INCLUDED
