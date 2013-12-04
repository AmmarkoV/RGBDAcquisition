#include <stdio.h>
#include <stdlib.h>

#include "Tools.h"


#include "../acquisitionSegment/AcquisitionSegment.h"
#include "../acquisition/Acquisition.h"

int XYOverRect(int x , int y , int rectx1,int recty1,int rectx2,int recty2)
{
  if ( (x>=rectx1) && (x<=rectx2) )
    {
      if ( (y>=recty1) && (y<=recty2) )
        {
          return 1;
        }
    }
  return 0;
}



int dumpCameraDepths(unsigned int moduleID , unsigned int devID , char * filename)
{
  FILE * fp=0;
  fp = fopen(filename,"w");
  if (fp!=0)
    {
      float x3D , y3D , z3D ;
      unsigned int x,y;
      for (y=0; y<480; y++)
      {
       for (x=0; x<640; x++)
       {
         if ( acquisitionGetDepth3DPointAtXYCameraSpace(moduleID,devID,x,y,&x3D,&y3D,&z3D) )
                  { fprintf(fp,"%0.4f %0.4f %0.4f ",x3D,y3D,z3D); }
       }
      }
      fclose(fp);
      return 1;
    }
  return 0;
}



int dumpExtDepths(unsigned int moduleID , unsigned int devID , char * filename)
{
  FILE * fp=0;
  fp = fopen(filename,"w");
  if (fp!=0)
    {
      float x3D , y3D , z3D ;
      unsigned int x,y;
      for (y=0; y<480; y++)
      {
       for (x=0; x<640; x++)
       {
         if ( acquisitionGetDepth3DPointAtXY(moduleID,devID,x,y,&x3D,&y3D,&z3D) )
                  { fprintf(fp,"%0.4f %0.4f %0.4f ",x3D,y3D,z3D); }
       }
      }
      fclose(fp);
      return 1;
    }
  return 0;
}
