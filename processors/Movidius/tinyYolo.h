#ifndef TINYYOLO_H_INCLUDED
#define TINYYOLO_H_INCLUDED

#include "MovidiusTypes.h"

int processTinyYOLO(struct labelContents * labels, float * results , unsigned int resultsLength ,char * pixels, unsigned int imageWidth, unsigned int imageHeight , float minimumConfidence);

#endif // TINYYOLO_H_INCLUDED
