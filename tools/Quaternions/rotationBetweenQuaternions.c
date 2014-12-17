#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <locale.h>

#include "../AmMatrix/quaternions.h"

int forceUSLocaleToKeepOurSanity()
{
   setlocale(LC_ALL, "en_US.UTF-8");
   setlocale(LC_NUMERIC, "en_US.UTF-8");
   return 1;
}

int main(int argc, char *argv[])
{
    forceUSLocaleToKeepOurSanity();
    if (argc<8) {
                  printf("RotationBetweenQuaternions quatX quatY quatZ quatW quatX quatY quatZ quatW you did not provide 8 arguments , you only provided %u\n",argc);
                  int i=0;
                  for (i=0; i<=argc; i++)
                  {
                      fprintf(stderr,"Arg %u = %s\n",i,argv[i]);
                  }
                  return 1;
                }

    double qA[4]; qA[0] = atof(argv[1]); qA[1] = atof(argv[2]); qA[2] = atof(argv[3]); qA[3] = atof(argv[4]);
    double qB[4]; qB[0] = atof(argv[5]); qB[1] = atof(argv[6]); qB[2] = atof(argv[7]); qB[3] = atof(argv[8]);


    if (
         (qA[0]==qB[0]) &&
         (qA[1]==qB[1]) &&
         (qA[2]==qB[2]) &&
         (qA[3]==qB[3])
       )
    {
      //This rarely (if ever) happens :P
      printf("0.0\n");
      return 0;
    }


    normalizeQuaternions(&qA[0],&qA[1],&qA[2],&qA[3]);
    normalizeQuaternions(&qB[0],&qB[1],&qB[2],&qB[3]);

    //fprintf(stderr,"quatA ( X %f , Y %f , Z %f , W %f ) , quatB ( X %f , Y %f , Z %f , W %f ) \n",qA[0],qA[1],qA[2],qA[3],qB[0],qB[1],qB[2],qB[3]);
    printf("%f\n",anglesBetweenQuaternions(qA[0],qA[1],qA[2],qA[3],qB[0],qB[1],qB[2],qB[3]));

    return 0;
}
