#include "calibration.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


#include "../../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/AmMatrix/matrix4x4Tools.h"
#include "../../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/AmMatrix/matrixCalculations.h"

#define DEBUG_PRINT_EACH_CALIBRATION_LINE_READ 0

#define DEFAULT_WIDTH 640
#define DEFAULT_HEIGHT 480

#define DEFAULT_FX 535.423874
#define DEFAULT_FY 533.484654

//Default Kinect things for reference
#define DEFAULT_FOCAL_LENGTH 120.0
#define DEFAULT_PIXEL_SIZE 0.1052

#define MAX_LINE_CALIBRATION 1024

int NullCalibration(unsigned int width,unsigned int height, struct calibration * calib)
{
  calib->width=width;
  calib->height=height;

  calib->intrinsicParametersSet=0;
  calib->extrinsicParametersSet=0;


  calib->nearPlane=0.1;
  calib->farPlane=100.0;

  calib->intrinsic[0]=0.0;  calib->intrinsic[1]=0.0;  calib->intrinsic[2]=0.0;
  calib->intrinsic[3]=0.0;  calib->intrinsic[4]=0.0;  calib->intrinsic[5]=0.0;
  calib->intrinsic[6]=0.0;  calib->intrinsic[7]=0.0;  calib->intrinsic[8]=1.0;

  calib->k1=0.0;  calib->k2=0.0; calib->p1=0.0; calib->p2=0.0; calib->k3=0.0;

  calib->extrinsicRotationRodriguez[0]=0.0; calib->extrinsicRotationRodriguez[1]=0.0; calib->extrinsicRotationRodriguez[2]=0.0;
  calib->extrinsicTranslation[0]=0.0; calib->extrinsicTranslation[1]=0.0; calib->extrinsicTranslation[2]=0.0;

  /*cx*/calib->intrinsic[CALIB_INTR_CX]  = (double) width/2;
  /*cy*/calib->intrinsic[CALIB_INTR_CY]  = (double) height/2;

  //-This is a bad initial estimation i guess :P
  /*fx*/ calib->intrinsic[CALIB_INTR_FX] = (double) (DEFAULT_FX * width) / DEFAULT_WIDTH;   //<- these might be wrong
  /*fy*/ calib->intrinsic[CALIB_INTR_FY] = (double) (DEFAULT_FY * height)  / DEFAULT_HEIGHT;    //<- these might be wrong
  //--------------------------------------------

  calib->depthUnit=1.0;

  return 1;
}

int ReadCalibration(char * filename,unsigned int width,unsigned int height,struct calibration * calib)
{
  //First free
  NullCalibration(width,height,calib);

  FILE * fp = 0;
  fp = fopen(filename,"r");
  if (fp == 0 ) {  return 0; }

  char line[MAX_LINE_CALIBRATION]={0};
  unsigned int lineLength=0;

  unsigned int i=0;

  unsigned int category=0;
  unsigned int linesAtCurrentCategory=0;


  while ( fgets(line,MAX_LINE_CALIBRATION,fp)!=0 )
   {
     unsigned int lineLength = strlen ( line );
     if ( lineLength > 0 ) {
                                 if (line[lineLength-1]==10) { line[lineLength-1]=0; /*fprintf(stderr,"-1 newline \n");*/ }
                                 if (line[lineLength-1]==13) { line[lineLength-1]=0; /*fprintf(stderr,"-1 newline \n");*/ }
                           }
     if ( lineLength > 1 ) {
                                 if (line[lineLength-2]==10) { line[lineLength-2]=0; /*fprintf(stderr,"-2 newline \n");*/ }
                                 if (line[lineLength-2]==13) { line[lineLength-2]=0; /*fprintf(stderr,"-2 newline \n");*/ }
                           }


     if (line[0]=='%') { linesAtCurrentCategory=0; }
     if ( (line[0]=='%') && (line[1]=='I') && (line[2]==0) )                   { category=1;    } else
     if ( (line[0]=='%') && (line[1]=='D') && (line[2]==0) )                   { category=2;    } else
     if ( (line[0]=='%') && (line[1]=='T') && (line[2]==0) )                   { category=3;    } else
     if ( (line[0]=='%') && (line[1]=='R') && (line[2]==0) )                   { category=4;    } else
     if ( (line[0]=='%') && (line[1]=='N') && (line[2]=='F') && (line[3]==0) ) { category=5;    } else
     if ( (line[0]=='%') && (line[1]=='U') && (line[2]=='N') && (line[3]=='I')
                         && (line[4]=='T') && (line[5]==0) )                   { category=6;    } else
        {
          #if DEBUG_PRINT_EACH_CALIBRATION_LINE_READ
           fprintf(stderr,"Line %u ( %s ) is category %u lines %u \n",i,line,category,linesAtCurrentCategory);
          #endif

          if (category==1)
          {
           calib->intrinsicParametersSet=1;
           switch(linesAtCurrentCategory)
           {
             case 1 :  calib->intrinsic[0] = atof(line); break;
             case 2 :  calib->intrinsic[1] = atof(line); break;
             case 3 :  calib->intrinsic[2] = atof(line); break;
             case 4 :  calib->intrinsic[3] = atof(line); break;
             case 5 :  calib->intrinsic[4] = atof(line); break;
             case 6 :  calib->intrinsic[5] = atof(line); break;
             case 7 :  calib->intrinsic[6] = atof(line); break;
             case 8 :  calib->intrinsic[7] = atof(line); break;
             case 9 :  calib->intrinsic[8] = atof(line); break;
           };
          } else
          if (category==2)
          {
           calib->intrinsicParametersSet=1;
           switch(linesAtCurrentCategory)
           {
             case 1 :  calib->k1 = atof(line); break;
             case 2 :  calib->k2 = atof(line); break;
             case 3 :  calib->p1 = atof(line); break;
             case 4 :  calib->p2 = atof(line); break;
             case 5 :  calib->k3 = atof(line); break;
           };
          } else
          if (category==3)
          {
           calib->extrinsicParametersSet=1;
           switch(linesAtCurrentCategory)
           {
             case 1 :  calib->extrinsicTranslation[0] = atof(line); break;
             case 2 :  calib->extrinsicTranslation[1] = atof(line); break;
             case 3 :  calib->extrinsicTranslation[2] = atof(line); break;
           };
          } else
          if (category==4)
          {
           calib->extrinsicParametersSet=1;
           switch(linesAtCurrentCategory)
           {
             case 1 :  calib->extrinsicRotationRodriguez[0] = atof(line); break;
             case 2 :  calib->extrinsicRotationRodriguez[1] = atof(line); break;
             case 3 :  calib->extrinsicRotationRodriguez[2] = atof(line); break;
           };
          }else
          if (category==5)
          {
           calib->extrinsicParametersSet=1;
           switch(linesAtCurrentCategory)
           {
             case 1 :  calib->nearPlane = atof(line); break;
             case 2 :  calib->farPlane  = atof(line); break;
           };
          }else
          if (category==6)
          {
           switch(linesAtCurrentCategory)
           {
             case 1 :  calib->depthUnit = atof(line); break;
           };
          }


        }

     ++linesAtCurrentCategory;
     ++i;
     line[0]=0;
   }

  return 1;
}







int WriteCalibration(char * filename,struct calibration * calib)
{
  FILE * fp = 0;
  fp = fopen(filename,"w");
  if (fp == 0 ) {  return 0; }

    fprintf( fp, "%%Calibration File\n");
    fprintf( fp, "%%CameraID=0\n");
    fprintf( fp, "%%CameraNo=0\n");

    time_t t;
    time( &t );
    struct tm *t2 = localtime( &t );
    char buf[1024];
    strftime( buf, sizeof(buf)-1, "%c", t2 );
    fprintf( fp, "%%Date=%s\n",buf);


    fprintf( fp, "%%ImageWidth=%u\n",calib->width);
    fprintf( fp, "%%ImageHeight=%u\n",calib->height);
    fprintf( fp, "%%Description=After %u images , board is %ux%u , square size is %u , aspect ratio %0.2f\n",
                                                    calib->imagesUsed,
                                                    calib->boardWidth,
                                                    calib->boardHeight,
                                                    calib->squareSize,
                                                    (double) calib->width/calib->height);


    fprintf( fp, "%%Intrinsics I[1,1], I[1,2], I[1,3], I[2,1], I[2,2], I[2,3], I[3,1], I[3,2] I[3,3] 3x3\n");
    fprintf( fp, "%%I\n");
    fprintf( fp, "%f\n",calib->intrinsic[0]); fprintf( fp, "%f\n",calib->intrinsic[1]); fprintf( fp, "%f\n",calib->intrinsic[2]);
    fprintf( fp, "%f\n",calib->intrinsic[3]); fprintf( fp, "%f\n",calib->intrinsic[4]); fprintf( fp, "%f\n",calib->intrinsic[5]);
    fprintf( fp, "%f\n",calib->intrinsic[6]); fprintf( fp, "%f\n",calib->intrinsic[7]); fprintf( fp, "%f\n",calib->intrinsic[8]);



    fprintf( fp, "%%Distortion D[1], D[2], D[3], D[4] D[5] \n");
    fprintf( fp, "%%D\n");
    fprintf( fp, "%f\n",calib->k1);
    fprintf( fp, "%f\n",calib->k2);
    fprintf( fp, "%f\n",calib->p1);
    fprintf( fp, "%f\n",calib->p2);
    fprintf( fp, "%f\n",calib->k3);

    if( calib->extrinsicParametersSet )
    {
      int i=0;
      for (i=0; i<1; i++)
      {
       fprintf( fp, "%%Translation T.X, T.Y, T.Z\n");
       fprintf( fp, "%%T\n");
       fprintf( fp, "%f\n",calib->extrinsicTranslation[0]);
       fprintf( fp, "%f\n",calib->extrinsicTranslation[1]);
       fprintf( fp, "%f\n",calib->extrinsicTranslation[2]);

       fprintf( fp, "%%Rotation Vector (Rodrigues) R.X, R.Y, R.Z \n");
       fprintf( fp, "%%R\n");
       fprintf( fp, "%f\n",calib->extrinsicRotationRodriguez[0]);
       fprintf( fp, "%f\n",calib->extrinsicRotationRodriguez[1]);
       fprintf( fp, "%f\n",calib->extrinsicRotationRodriguez[2]);
      }
     }

 return 1;
}


int transform3DPointUsingCalibration(struct calibration * calib , float * x , float * y , float * z)
{
 double * m = alloc4x4Matrix();
 if (m==0) {fprintf(stderr,"Could not allocate a 4x4 matrix , cannot perform bounding box selection\n"); } else
 {
  create4x4IdentityMatrix(m);
  if ( calib->extrinsicParametersSet )
     { convertRodriguezAndTranslationTo4x4DMatrix(m, calib->extrinsicRotationRodriguez , calib->extrinsicTranslation); }
     else
     {fprintf(stderr,"No extrinsic parameters provided , bounding box segmentation will use default coordinate system \n"); }

  double raw3D[4]={0};
  double world3D[4]={0};

  raw3D[0] = (double) *x;
  raw3D[1] = (double) *y;
  raw3D[2] = (double) *z;
  raw3D[3] = (double) 1.0;

  transform3DPointUsing4x4Matrix(world3D,m,raw3D);

  free4x4Matrix(&m); // This is the same as free(m); m=0;
  return 1;
 } //End of M allocated!

  return 0;
}




