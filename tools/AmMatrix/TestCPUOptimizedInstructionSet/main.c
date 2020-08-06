#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../matrix4x4Tools.h"

unsigned long tickBaseMN=0;

unsigned long GetTickCountMicrosecondsMN()
{
    struct timespec ts;
    if ( clock_gettime(CLOCK_MONOTONIC,&ts) != 0)
    {
        return 0;
    }

    if (tickBaseMN==0)
    {
        tickBaseMN = ts.tv_sec*1000000 + ts.tv_nsec/1000;
        return 0;
    }

    return ( ts.tv_sec*1000000 + ts.tv_nsec/1000 ) - tickBaseMN;
}


int main()
{
    if (!codeHasSSE())
    {
      printf("SSE code is not compiled in..\n");
      exit(0);
    }

    int i=0;
    float __attribute__((aligned(16))) testResultOptimized[16]= {0};
    float testResultUnoptimized[16]= {0};
    float __attribute__((aligned(16))) matrixA[16]= {0};
    float __attribute__((aligned(16))) matrixB[16]= {0};

    unsigned int numberOfSamples = 1000;
    unsigned long unoptimizedTime = 0;
    unsigned long optimizedTime = 0;

    for (i=0; i<numberOfSamples; i++)
    {
        unsigned long startUnoptimized = GetTickCountMicrosecondsMN();
        multiplyTwo4x4FMatrices(testResultUnoptimized,matrixA,matrixB);
        unsigned long endUnoptimized = GetTickCountMicrosecondsMN();
        unoptimizedTime+=endUnoptimized - startUnoptimized;

        unsigned long startOptimized = GetTickCountMicrosecondsMN();
        multiplyTwo4x4FMatrices_SSE(testResultOptimized,matrixA,matrixB);
        unsigned long endOptimized = GetTickCountMicrosecondsMN();
        optimizedTime+=endOptimized - startOptimized;
    }
    printf("%0.4f microseconds unoptimized!\n",(float) unoptimizedTime/numberOfSamples);
    printf("%0.4f microseconds optimized!\n",(float) optimizedTime/numberOfSamples);
    return 0;
}
