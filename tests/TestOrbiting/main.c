#include <stdio.h>
#include <stdlib.h>
#include "../../tools/AmMatrix/matrixCalculations.h"





int generatePlot(unsigned int viewX,unsigned int viewY)
{
  char command[1024]={0};
  //-----------------------
  snprintf(command,1024,"gnuplot -e 'set terminal png; set output \"plot_%u_%u.png\"; set title \"3D random points\"; set view %u,%u; splot \"output.dat\"  using 1:2:3:3 with points palette'",viewX,viewY,viewX,viewY);
  int  i=system(command);
  if (i!=0) { fprintf(stderr,"Error generating graph ( is gnuplot installed ? )"); }

  //-----------------------
  snprintf(command,1024,"timeout 10 gpicview  \"plot_%u_%u.png\"& ",viewX,viewY);
  i=system(command);
  if (i!=0) { fprintf(stderr,"Error generating graph ( is gpicview installed ? )"); }

 return 1;
}



int main()
{
    fprintf(stderr,"\n\n\nLets Test Orbiting..!\n");

    double xOffset=0.0,yOffset=0.0,zOffset=00.0;


    double satPosAbsolute[4]={0};
    double planetPosAbsolute[4]={0};
    double planetRotAbsolute[4]={0};
    //
    double satPosRelative[4]={0};


    planetPosAbsolute[0]=400.0;
    planetPosAbsolute[1]=400.0;
    planetPosAbsolute[2]=400.0;
    planetPosAbsolute[3]=1.0;


    planetRotAbsolute[0]=0.0;
    planetRotAbsolute[1]=0.0;
    planetRotAbsolute[2]=0.0;


    satPosAbsolute[0]=1000.0;
    satPosAbsolute[1]=1000.0;
    satPosAbsolute[2]=1000.0;
    satPosAbsolute[3]=1.0;

    fprintf(stderr,"Sattelite Pos Absolute %0.2f %0.2f %0.2f \n",satPosAbsolute[0],satPosAbsolute[1],satPosAbsolute[2]);

    pointFromAbsoluteToRelationWithObject_PosXYZRotationXYZ(satPosRelative,planetPosAbsolute,planetRotAbsolute,satPosAbsolute);

    fprintf(stderr,"   Planet Pos : %0.2f %0.2f %0.2f \n",planetPosAbsolute[0],planetPosAbsolute[1],planetPosAbsolute[2]);
    fprintf(stderr,"   Planet Rot : %0.2f %0.2f %0.2f \n",planetRotAbsolute[0],planetRotAbsolute[1],planetRotAbsolute[2]);

    fprintf(stderr,"   Sattelite Pos Relative %0.2f %0.2f %0.2f \n",satPosRelative[0],satPosRelative[1],satPosRelative[2]);


  FILE * pFile  = fopen ("output.dat","w");
  fprintf(pFile,"%0.2f %0.2f %0.2f\n",planetPosAbsolute[0],planetPosAbsolute[1],planetPosAbsolute[2]);


  if (pFile!=NULL)
  {
    fprintf(pFile,"%0.2f %0.2f %0.2f\n",satPosAbsolute[0],satPosAbsolute[1],satPosAbsolute[2]);
    unsigned int i=0;

    for (i=0; i<180; i++)
    {
     planetRotAbsolute[0]=xOffset+i*2.0; planetRotAbsolute[1]=yOffset+0.0; planetRotAbsolute[2]=zOffset+0.0;
     if ( pointFromRelationWithObjectToAbsolute_PosXYZRotationXYZ(satPosAbsolute,planetPosAbsolute,planetRotAbsolute,satPosRelative) )
       {
         fprintf(pFile,"%0.2f %0.2f %0.2f\n",satPosAbsolute[0],satPosAbsolute[1],satPosAbsolute[2]);
       }
    }

   for (i=0; i<180; i++)
    {
     planetRotAbsolute[0]=xOffset+0; planetRotAbsolute[1]=yOffset+i*2.0; planetRotAbsolute[2]=zOffset+0.0;
     if ( pointFromRelationWithObjectToAbsolute_PosXYZRotationXYZ(satPosAbsolute,planetPosAbsolute,planetRotAbsolute,satPosRelative) )
       {
         fprintf(pFile,"%0.2f %0.2f %0.2f\n",satPosAbsolute[0],satPosAbsolute[1],satPosAbsolute[2]);
       }
    }



   for (i=0; i<180; i++)
    {
     planetRotAbsolute[0]=xOffset+0; planetRotAbsolute[1]=yOffset+0.0; planetRotAbsolute[2]=zOffset+i*2.0;
     if ( pointFromRelationWithObjectToAbsolute_PosXYZRotationXYZ(satPosAbsolute,planetPosAbsolute,planetRotAbsolute,satPosRelative) )
       {
         fprintf(pFile,"%0.2f %0.2f %0.2f\n",satPosAbsolute[0],satPosAbsolute[1],satPosAbsolute[2]);
       }
    }

    fclose (pFile);





    generatePlot(0,0);
    generatePlot(45,0);
    generatePlot(90,0);

    generatePlot(45,45);



  }


    return 0;
}
