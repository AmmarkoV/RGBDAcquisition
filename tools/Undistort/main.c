#include <stdio.h>
#include <stdlib.h>

#include "../Codecs/codecs.h"


struct pointToAdd
{
    unsigned int source;
    unsigned int target;
};

struct resectionData
{
  unsigned int * directMapping;

  unsigned int MAXPoints;
  unsigned int curPoints;
  struct pointToAdd * pointsListThatNeedInterpolation;
};


int randNeigborFillForHolesInResectioning( struct resectionData * res , unsigned int width , unsigned int height)
{
 unsigned int * M = res->directMapping;
 unsigned int holes=0,filled=0,totalHoles=0;
 unsigned int i=0,x=0,y=0;
 unsigned int memLimit = width * height ;


 //This should populate res->pointsListThatNeedInterpolation
 //For now i am just trying to fill some holes with stupid uninterpolated values

 //We try to unfold the directMapping on an RGB out buffer and see what is missing so that we can later fill all the holes
 unsigned char * output = (unsigned char *) malloc( sizeof(unsigned char) * memLimit * 3);
 if (output==0) { fprintf(stderr,"Could not allocate memory for finding out holes\n"); return 0; }
 unsigned int ptr=0 , oldSource = 0, newTarget = 0, ptr_end =  memLimit , M_Ptr = 0;
 memset(output, 0 , memLimit * 3 *sizeof(unsigned char));
 for (ptr=0; ptr<ptr_end; ptr++)
    {
       M_Ptr = M[ptr];
       if ( (M_Ptr!=0) && (M_Ptr< memLimit*3 ) )  { output[M_Ptr] = 1;  output[M_Ptr+1] = 1;  output[M_Ptr+2] = 1;}
    }
 //Ok after that all pixels marked as 1 are ok , all pixels marked with zero need filling

 for (i=0; i< memLimit*3; i+=3) { if (output[i]==0) { ++totalHoles;  } }
 fprintf(stderr,"Unfolding reveals %u holes \n", totalHoles);


 res->MAXPoints=totalHoles;
 res->curPoints=0;
 res->pointsListThatNeedInterpolation = (struct pointToAdd *) malloc(sizeof (struct pointToAdd)  * (totalHoles+1) );
 if (res->pointsListThatNeedInterpolation==0) { fprintf(stderr,"Could not allocate memory for %u points added \n",totalHoles); return 0; }
 struct pointToAdd * newPoint = res->pointsListThatNeedInterpolation ;





  /*
      A B C   A is i-width , B is i-1-width , C is i+1-width
      D E F   E is i       , D is i-1       , F is i+1
      G H I   G is i+width , H is i-1+width , I is i+1+width
  */

 unsigned int convElements[9]={0};


for (y=1; y<height-1; y++)
 {
   i=y * width *3 ;
   for (x=1; x<width-1; x++)
   {
     i+=3;

     convElements[4]=i-3; convElements[5]=i; convElements[6]=i+3;
     convElements[1]=convElements[4]-(width*3); convElements[2]=convElements[5]-(width*3); convElements[3]=convElements[6]-(width*3);
     convElements[7]=convElements[4]+(width*3); convElements[8]=convElements[5]+(width*3); convElements[9]=convElements[6]+(width*3);

     if (res->curPoints>=totalHoles)
     {
       fprintf(stderr,"!"); //This should be impossible to happen :P , since we've already counted our holes and we know exactly how many there are
       //but it may happen and this is the way to protect our program from segfaulting if it does :P
     } else
     if (output[i]==0)
       {
         ++holes;
         if (output[convElements[0]]!=0)
          { newPoint[res->curPoints].source = convElements[0]; newPoint[res->curPoints].target = i; ++res->curPoints;  output[i]=1; } else
         if (output[convElements[1]]!=0)
          { newPoint[res->curPoints].source = convElements[1]; newPoint[res->curPoints].target = i; ++res->curPoints;  output[i]=1; } else
         if (output[convElements[2]]!=0)
          { newPoint[res->curPoints].source = convElements[2]; newPoint[res->curPoints].target = i; ++res->curPoints;  output[i]=1; } else

         if (output[convElements[3]]!=0)
          { newPoint[res->curPoints].source = convElements[3]; newPoint[res->curPoints].target = i; ++res->curPoints;  output[i]=1; } else
         if (output[convElements[5]]!=0)
          { newPoint[res->curPoints].source = convElements[5]; newPoint[res->curPoints].target = i; ++res->curPoints;  output[i]=1; } else

         if (output[convElements[6]]!=0)
          { newPoint[res->curPoints].source = convElements[6]; newPoint[res->curPoints].target = i; ++res->curPoints;  output[i]=1; } else
         if (output[convElements[7]]!=0)
          { newPoint[res->curPoints].source = convElements[7]; newPoint[res->curPoints].target = i; ++res->curPoints;  output[i]=1; } else
         if (output[convElements[8]]!=0)
          { newPoint[res->curPoints].source = convElements[8]; newPoint[res->curPoints].target = i; ++res->curPoints;  output[i]=1; }
       }
   }
 }

 //Count Holes one last time!
 //There should be some on the edges of the frame , can't really do much about them :S
 totalHoles=0;
 for (i=0; i< memLimit*3; i+=3) { if (output[i]==0) { ++totalHoles;  } }
 fprintf(stderr,"Final unfolding reveals %u holes \n", totalHoles);

 free(output);
 return 1;
}



/*

   RectifiedPoint = M * OriginalPoint

   Rectified Point = [ new_x new_y new_w ]
   Original Point  = [ x y w ]

         |fx  0   cx|       a   b   c
   M =   |0   fy  cy|       d   e   f
         |0   0   1 |       g   h   i
*/
struct resectionData * precalculateResectioning( unsigned int width , unsigned int height,
                                                                    double fx,double fy , double cx,double cy ,
                                                                    double k1,double k2 , double p1,double p2 , double k3 )
{
   struct resectionData * res = (struct resectionData *) malloc (sizeof(struct resectionData));
   if (res==0) { fprintf(stderr,"Could not allocate memory for resectioning structure\n"); return 0; }
   res->directMapping = (unsigned int *) malloc (width * height * sizeof(unsigned int));
   if (res->directMapping==0) { fprintf(stderr,"Could not allocate memory for resectioning structure\n"); return 0; }
   memset(res->directMapping,0,width * height * sizeof(unsigned int));
   res->pointsListThatNeedInterpolation = 0;

  unsigned int * frame = res->directMapping;

  fprintf(stderr,"Calibrating for fx=%f fy=%f cx=%f cy=%f\n",fx,fy,cx,cy);
  fprintf(stderr,"k1=%f k2=%f p1=%f p2=%f k3=%f \n",k1,k2,p1,p2,k3);

  if ( ( fx == 0.0) || ( fy == 0.0) || ( (k1==0.0)&&(k2==0.0)&&(k3==0.0) )) { fprintf(stderr,"Erroneous parameters calibration canceled\n"); return 0; }

  unsigned int i,x = width ,y= height , mem , new_mem , addressForErrors;
  unsigned int undistorted_x,undistorted_y;

  mem = 0;
  double ifx=1.f/fx,ify=1.f/fy;
  double dstdx,dstdy , distx,disty;
  double dx,dy;
  double r_sq = 0; // R Squared
  double r_cu = 0; // R Cubed
  double k_coefficient = 0;
  double new_x,new_y;

  // SEE http://opencv.willowgarage.com/documentation/camera_calibration_and_3d_reconstruction.html
  // https://code.ros.org/trac/opencv/browser/trunk/opencv/src/cv/cvundistort.cpp?rev=18
  // https://code.ros.org/trac/opencv/browser/trunk/opencv/modules/imgproc/src/undistort.cpp?rev=4885
  // http://tech.groups.yahoo.com/group/OpenCV/message/26019
  // Also Learning OpenCV page 375-377
  /*

Finaly fixed using code from Philip Gruebele @
http://tech.groups.yahoo.com/group/OpenCV/message/26019

archived at 3dpartylibs/code/undistort_point.cpp
*/
  unsigned int PrecisionErrors=0;
  unsigned int OffFrame=0;


  for (y=0; y<height; y++)
  {
     addressForErrors=0;
     for (x=0; x<width; x++)
        {
          //Well this is supposed to rectify lens distortions based on calibration done with my image sets
          //but the values returned are way off ..
          dstdx = ( x - cx );
          dstdx *= ifx;

          dstdy = ( y - cy );
          dstdy *= ify;

          new_x = dstdx;
          new_y = dstdy;

          for ( i=0; i<5; i++)
           {
               r_sq = new_x*new_x;
               r_sq += new_y*new_y;

               r_cu = r_sq*r_sq;

               k_coefficient = 1;
               k_coefficient += k1 * r_sq;
               k_coefficient += k2 * r_cu;
               k_coefficient += k3 * r_cu * r_sq ;

               dx = 2 * p1 * new_x * new_y;
               dx += p2 * ( r_sq + 2 * new_x * new_x);

               dy = 2 * p2 * new_x * new_y;
               dy += p1 * ( r_sq + 2 * new_y * new_y);

               new_x = ( dstdx - dx );
               new_x /= k_coefficient;

               new_y = ( dstdy - dy );
               new_y /= k_coefficient;
           }

          dstdx = new_x;
          dstdy = new_y;

          dstdx *= fx; dstdx += cx;
          dstdy *= fy; dstdy += cy;


          undistorted_x = (unsigned int) round(dstdx);
          undistorted_y = (unsigned int) round(dstdy);



                   /* REVERSE CHECK ! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>*/
                   new_x = dstdx - cx; new_x /= fx;
                   new_y = dstdy - cy; new_y /= fy;

                   r_sq = new_x*new_x + new_y*new_y;
                   distx = new_x ;
                   distx += new_x*(k1*r_sq + k2*r_sq*r_sq);
                   distx += (2*p1*new_x*new_y + p2*(r_sq + 2*new_x*new_x));


                   disty = new_y;
                   disty +=new_y*(k1*r_sq + k2*r_sq*r_sq);
                   disty +=(2*p2*new_x*new_y + p1*(r_sq + 2*new_y*new_y));


                   distx *= fx; distx += cx;
                   disty *= fy; disty += cy;
                   /* <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<*/



      // This should never be more than .2 pixels...
      double diffx = x - distx;
      double diffy = y - disty;

         if ( (diffx> 0.2) || (diffy>0.2) )
          {
             /* ACCURACY ERROR , This means that we have a percision error in the way math is done*/
             //fprintf(stderr,"$%u,%u to %u,%u",x,y,undistorted_x,undistorted_y);
             new_mem = 0;
                 new_mem=addressForErrors ; //TEST THIS USESTHE INTERPOLATED TO GET RID OF SOME BLANK SPOT ARTIFACTS
             ++PrecisionErrors;
          }

          if ( ( undistorted_x >= width ) || ( undistorted_y >= height ) )
             {
               // OFF RESULTS SHOULD BE INTERPOLATED WITH CLOSE MEMORY SPOTS
               new_mem=addressForErrors ; //TEST THIS USESTHE INTERPOLATED TO GET RID OF SOME BLANK SPOT ARTIFACTS
               ++OffFrame;
             } else
             {
                //We need the memory to point to the ->RGB address of the value
                new_mem = (undistorted_y * width * 3) + undistorted_x * 3 ;
                //fprintf(stderr,"%u,%u -> %u,%u .. \n",x,y,undistorted_x,undistorted_y);
             }

          if(new_mem!=0) { addressForErrors=new_mem; }
          frame [mem] = new_mem;
          ++mem;

       } /*END OF X LOOP*/
   }

 int totalHoles =0 ;
 for (x=0; x<height*width; x++) { if (frame[x]==0) { ++totalHoles; } }

 fprintf(stderr,"PrecalculationErrors -Total %u ,  Precision=%u , OffFrame=%u\n",totalHoles,PrecisionErrors,OffFrame);
 randNeigborFillForHolesInResectioning(res,width,height);

 totalHoles =0 ;
 for (x=0; x<height*width; x++) { if (frame[x]==0) { ++totalHoles; } }
 fprintf(stderr,"After interpolation Total %u \n",totalHoles);

 return res;
}

int freeResectioning( struct resectionData * res)
{
  if (res!=0)
  {
    if (res->directMapping!=0 )  { free(res->directMapping); res->directMapping=0; }
    if (res->pointsListThatNeedInterpolation!=0 )  { free(res->pointsListThatNeedInterpolation); res->pointsListThatNeedInterpolation=0; }
    free(res);
    res = 0;
  }
  return 1;
}

int undistortImage(unsigned char * input , unsigned char * output , unsigned int width , unsigned int height , struct resectionData * res)
{
 if ( (input==0) || (output==0) ) { fprintf(stderr,"Cannot undistortImage , image is null\n"); return 0; }
 if ( (width==0) || (height==0) ) { fprintf(stderr,"Cannot undistortImage , image input/output has null dimensions\n"); return 0; }
 if ( res==0) { fprintf(stderr,"Cannot undistortImage , resectionData is null\n"); return 0; }

 unsigned int memLimit = width * height ;
 unsigned int * M = res->directMapping;
 unsigned int ptr=0 , oldSource = 0, newTarget = 0, ptr_end = memLimit;

 unsigned int ClearValue=0;//255; //0 filters out empty things , 255 is white so they can be seen and debugged :P
 memset(output,ClearValue,memLimit*3*sizeof(unsigned char));


 for (ptr=0; ptr<ptr_end; ptr++)
 {
  newTarget = M[ptr];
  oldSource  = ptr*3;
  output[newTarget] = input[oldSource]; //Move R
  ++newTarget; ++oldSource;
  output[newTarget] = input[oldSource]; //Move G
  ++newTarget; ++oldSource ;
  output[newTarget] = input[oldSource]; //Move B
 }

 fprintf(stderr,"also undistorting %u new points \n",res->curPoints);
 for (ptr=0; ptr<res->curPoints; ptr++)
 {
  newTarget = res->pointsListThatNeedInterpolation[ptr].target;
  oldSource = res->pointsListThatNeedInterpolation[ptr].source;
  output[newTarget] = input[oldSource]; //Move R
  ++newTarget; ++oldSource;
  output[newTarget] = input[oldSource]; //Move G
  ++newTarget; ++oldSource ;
  output[newTarget] = input[oldSource]; //Move B
 }






return 1;
}


int main(int argc, char** argv)
{
   if( argc < 12)
   {
     fprintf(stderr,"Provided %u arguments\n",argc);
     printf(" Usage: undistort ImageToLoadAndUndistort Output fx fy cx cy k1 k2 p1 p2 k3\n");
     return 1;
   }

  struct Image * img  = readImage(argv[1],PPM_CODEC,0);
  if (img==0) { fprintf(stderr,"Could not open file %s \n",argv[1]); return 1; }

  double fx = 535.784106 , fy = 534.223354 , cx = 312.428312 , cy = 243.889369;
  double k1 = 0.021026 , k2 = -0.069355 , p1 = 0.000598 , p2 = 0.001729 , k3 = 0.0;

  struct resectionData *   res = precalculateResectioning(img->width,img->height,fx,fy,cx,cy,k1,k2,p1,p2,k3);
  if (res==0) { fprintf(stderr,"Could not generate resection data for file %s \n",argv[1]); return 1; }

  struct Image * undistortedImg = createImage(img->width,img->height , img->channels , img->bitsperpixel );
  if (undistortedImg==0) { fprintf(stderr,"Could not generate output image file to hold %s \n",argv[2]); return 1; }


  undistortImage(img->pixels, undistortedImg->pixels , img->width , img->height , res );

  writeImageFile(undistortedImg,PPM_CODEC,argv[2]);

  destroyImage(img);
  destroyImage(undistortedImg);


  freeResectioning(res);

  printf("done.. \n");

  return 0;
}
