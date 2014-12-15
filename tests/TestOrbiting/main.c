#include <stdio.h>
#include <stdlib.h>
#include "../../tools/AmMatrix/matrixCalculations.h"



int affixSatteliteToPlanetFromFrameForLength(
                                              double * satPosAbsolute ,
                                              double * planetPosAbsolute ,
                                              double * planetRotAbsolute,
                                              double * satPosRelative
                                             )
{
    //There is literally no good reason to go from rotation -> quaternion -> 3x3 -> quaternion -> rotation this could be optimized
    //==================================================================================

 /*
    unsigned int pos=0;
    for (pos=frameNumber+1; pos<frameNumber+duration; pos++)
    {
       planetPosAbsolute[0] = (double) stream->object[planetObj].frame[pos].x;
       planetPosAbsolute[1] = (double) stream->object[planetObj].frame[pos].y;
       planetPosAbsolute[2] = (double) stream->object[planetObj].frame[pos].z;
       planetPosAbsolute[3] = 1.0;

       planetRotAbsolute[0] = (double) stream->object[planetObj].frame[pos].rot1;
       planetRotAbsolute[1] = (double) stream->object[planetObj].frame[pos].rot2;
       planetRotAbsolute[2] = (double) stream->object[planetObj].frame[pos].rot3;

       if ( pointFromRelationWithObjectToAbsolute_PosXYZRotationXYZ(satPosAbsolute,planetPosAbsolute,planetRotAbsolute,satPosRelative) )
       {
           stream->object[satteliteObj].frame[pos].x = (float) satPosAbsolute[0];
           stream->object[satteliteObj].frame[pos].y = (float) satPosAbsolute[1];
           stream->object[satteliteObj].frame[pos].z = (float) satPosAbsolute[2];
       }
    }*/
 return 1;

}




int main()
{
    printf("Lets Test Orbiting..!\n");

    double satPosAbsolute[4]={0};
    double planetPosAbsolute[4]={0};
    double planetRotAbsolute[4]={0};
    double satPosRelative[4]={0};


    planetPosAbsolute[0]=0.0;
    planetPosAbsolute[1]=0.0;
    planetPosAbsolute[2]=0.0;
    planetPosAbsolute[3]=1.0;


    planetRotAbsolute[0]=0.0;
    planetRotAbsolute[1]=0.0;
    planetRotAbsolute[2]=0.0;


    satPosAbsolute[0]=10.0;
    satPosAbsolute[1]=10.0;
    satPosAbsolute[2]=10.0;
    satPosAbsolute[3]=1.0;

    fprintf(stderr,"Sattelite Pos Absolute %0.2f %0.2f %0.2f \n",satPosAbsolute[0],satPosAbsolute[1],satPosAbsolute[2]);

    pointFromAbsoluteToRelationWithObject_PosXYZRotationXYZ(1,satPosRelative,planetPosAbsolute,planetRotAbsolute,satPosAbsolute);

    fprintf(stderr,"Sattelite Pos Relative %0.2f %0.2f %0.2f \n",satPosRelative[0],satPosRelative[1],satPosRelative[2]);



    return 0;
}
