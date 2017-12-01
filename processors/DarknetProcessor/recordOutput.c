#include "recordOutput.h"

#include <stdio.h>
#include <stdlib.h>


int logEvent(
              unsigned int frameNumber,
              float x,
              float y,
              float width,
              float height,
              const char * label,
              float probability
            )
{
   fprintf(stderr,"detection(%u,%s,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f",
           frameNumber,
           label,
           probability,
           x,
           y,
           width,
           height
          );
   return 1;
}
