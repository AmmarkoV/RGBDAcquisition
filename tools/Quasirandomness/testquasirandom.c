//gcc testquasirandom.c libQuasirandomness.a -o test

#include <stdio.h>
#include <stdlib.h>
#include "quasirandomness.h" 

int main(int argc, char *argv[])
{
  struct quasiRandomizerContext qrc;
  initializeQuasirandomnessContext(&qrc,640,640,640,1);
  
  FILE *fp = 0;
  fp = fopen("quasi.dat","w");
  if (fp!=0)
  {  
    float x,y,z;
    unsigned int i=0;
    for (i=0; i<100; i++)
     {
      getNextRandomPoint(&qrc,&x,&y,&z);    
      fprintf(fp,"%f %f %f\n",x,y,z); 
     }
    fclose(fp);
  } 

int i=system("gnuplot -e 'set terminal png; set output \"quasi.png\"; set title \"3D random points\"; splot \"quasi.dat\" using 1:2:3:3 with points palette'");



}
