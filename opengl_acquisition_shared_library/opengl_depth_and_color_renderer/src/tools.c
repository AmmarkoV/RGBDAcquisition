#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <GL/gl.h>

#include "tools.h"


int checkOpenGLError(char * file , int  line)
{
  int err=glGetError();
  if (err !=  GL_NO_ERROR /*0*/ ) {  fprintf(stderr,"OpenGL Error (%u) : %s %u \n ", err , file ,line ); return 1; }
 return 0;
}


char * loadFileToMem(char * filename,unsigned long * file_length)
{
  if (filename==0)  { fprintf(stderr,"Could not load shader incorrect filename \n"); return 0; }
  if (file_length==0)  { fprintf(stderr,"Could not load shader %s , incorrect file length parameter \n",filename); return 0; }

  FILE * pFile;
  long lSize;
  char * buffer;
  size_t result;

  pFile = fopen ( filename , "rb" );
  if (pFile==0) { fprintf(stderr,"Could not open shader file %s \n",filename); return 0;}

  // obtain file size :
  fseek (pFile , 0 , SEEK_END);
  lSize = ftell (pFile);
  rewind (pFile);

  // allocate memory to contain the whole file:
  buffer = (char*) malloc ( (sizeof(char)*lSize)+1 );
  if (buffer == 0) {fprintf(stderr,"Could not allocate %u bytes of memory for shader file %s \n",(unsigned int ) lSize,filename); return 0; }

  // copy the file into the buffer:
  result = fread (buffer,1,lSize,pFile);
  if (result != lSize) {fputs ("Reading error",stderr); free(buffer); return 0; }

  /* the whole file is now loaded in the memory buffer. */

  // terminate
  fclose (pFile);

  buffer[lSize]=0; //Add a null termination for shader usage
  *file_length = lSize;
  return buffer;
}


/*
  TAKEN FROM http://www.lighthouse3d.com/opengl/maths/index.php?raytriint

*/


#define innerProduct(v,q) \
       ((v)[0] * (q)[0] + \
		(v)[1] * (q)[1] + \
		(v)[2] * (q)[2])


#define crossProduct(a,b,c) \
        (a)[0] = (b)[1] * (c)[2] - (c)[1] * (b)[2]; \
        (a)[1] = (b)[2] * (c)[0] - (c)[2] * (b)[0]; \
        (a)[2] = (b)[0] * (c)[1] - (c)[0] * (b)[1];



/* a = b - c */
#define vector(a,b,c) \
        (a)[0] = (b)[0] - (c)[0];	\
        (a)[1] = (b)[1] - (c)[1];	\
        (a)[2] = (b)[2] - (c)[2];


int rayIntersectsTriangle(float *p, float *d,float *v0, float *v1, float *v2)
{

	float e1[3],e2[3],h[3],s[3],q[3];
	float a,f,u,v;

	vector(e1,v1,v0);
	vector(e2,v2,v0);
	crossProduct(h,d,e2);
	a = innerProduct(e1,h);

	if (a > -0.00001 && a < 0.00001)
		return(0);

	f = 1/a;
	vector(s,p,v0);
	u = f * (innerProduct(s,h));

	if (u < 0.0 || u > 1.0)
		return(0);

	crossProduct(q,s,e1);
	v = f * innerProduct(d,q);
	if (v < 0.0 || u + v > 1.0)
		return(0);
	// at this stage we can compute t to find out where
	// the intersection point is on the line
	float t = f * innerProduct(e2,q);
	if (t > 0.00001) // ray intersection
		return(1);
	else // this means that there is a line intersection
		 // but not a ray intersection
		 return (0);
}


int rayIntersectsRectangle(float *p, float *d,float *v0, float *v1, float *v2, float *v3)
{
   if (  rayIntersectsTriangle(p,d,v0,v1,v2) )
     {
       return 1;
     }

   if (  rayIntersectsTriangle(p,d,v1,v2,v3) )
     {
       return 1;
     }

   return 0;
}

//http://ilab.usc.edu/wiki/index.php/Fast_Square_Root
inline float sqrt_fast_approximation(const float x)
{
  union
  {
    int i;
    float x;
  } u;

  u.x = x;
  u.i = (1<<29) + (u.i >> 1) - (1<<22);
  return u.x;
}




double distanceBetween3DPoints(double * p1, double * p2)
{
  double x1 = p1[0] , y1 = p1[1] , z1 = p1[2];
  double x2 = p2[0] , y2 = p2[1] , z2 = p2[2];

  double dx=0.0,dy=0.0,dz=0.0;

  //I Could actually skip this
  if (x1>=x2) { dx=x1-x2; } else { dx=x2-x1; }
  if (y1>=y2) { dy=y1-y2; } else { dy=y2-y1; }
  if (z1>=z2) { dz=z1-z2; } else { dz=z2-z1; }
  //==========================

  return (double) sqrt( (dx * dx) + (dy * dy) + (dz * dz) );
}


float distanceBetween3DPointsFast(float *x1,float*y1,float *z1,float *x2,float*y2,float *z2)
{
    //sqrt_fast_approximation
  float dx,dy,dz;

  if (*x1>=*x2) { dx=*x1-*x2; } else { dx=*x2-*x1; }
  if (*y1>=*y2) { dy=*y1-*y2; } else { dy=*y2-*y1; }
  if (*z1>=*z2) { dz=*z1-*z2; } else { dz=*z2-*z1; }

  return (float) sqrt_fast_approximation( (dx * dx) + (dy * dy) + (dz * dz) );
}

float squaredDistanceBetween3DPoints(float *x1,float*y1,float *z1,float *x2,float*y2,float *z2)
{
  float dx,dy,dz;

  if (*x1>=*x2) { dx=*x1-*x2; } else { dx=*x2-*x1; }
  if (*y1>=*y2) { dy=*y1-*y2; } else { dy=*y2-*y1; }
  if (*z1>=*z2) { dz=*z1-*z2; } else { dz=*z2-*z1; }

  return (float)  (dx * dx) + (dy * dy) + (dz * dz) ;
}


float RGB2OGL(unsigned int colr)
{
  return (float) colr/255;
}

float calculateDistance(float from_x,float from_y,float from_z,float to_x,float to_y,float to_z)
{
   float vect_x = from_x - to_x;
   float vect_y = from_y - to_y;
   float vect_z = from_z - to_z;

   return  (sqrt(pow(vect_x, 2) + pow(vect_y, 2) + pow(vect_z, 2)));

}


void vectorDirection(float src_x,float src_y,float src_z,float targ_x,float targ_y,float targ_z,float *vect_x,float *vect_y,float *vect_z)
{
    *vect_x = src_x - targ_x;
    *vect_y = src_y - targ_y;
    *vect_z = src_z - targ_z;

    float len = (sqrt(pow(*vect_x, 2) + pow(*vect_y, 2) + pow(*vect_z, 2)));
    if(len == 0) len = 1.0f;

    *vect_x /= len ;
    *vect_y /= len ;
    *vect_z /= len ;
}



void findNormal(float *v1x, float *v1y, float *v1z, float v2x, float v2y, float v2z, float v3x, float v3y, float v3z )
{ char x = 1;
  char y = 2;
  char z = 3;
  float temp_v1[3];
  float temp_v2[3];
  float temp_lenght;
  float CNormal[3];

temp_v1[x] = *v1x - v2x;
temp_v1[y] = *v1y - v2y;
temp_v1[z] = *v1z - v2z;

temp_v2[x] = v2x - v3x;
temp_v2[y] = v2y - v3y;
temp_v2[z] = v2z - v3z;

// calculate cross product
CNormal[x] = temp_v1[y]*temp_v2[z] - temp_v1[z]*temp_v2[y];
CNormal[y] = temp_v1[z]*temp_v2[x] - temp_v1[x]*temp_v2[z];
CNormal[z] = temp_v1[x]*temp_v2[y] - temp_v1[y]*temp_v2[x];

// normalize normal
temp_lenght =(CNormal[x]*CNormal[x])+ (CNormal[y]*CNormal[y])+ (CNormal[z]*CNormal[z]);

temp_lenght = sqrt(temp_lenght);

// prevent n/0
if (temp_lenght == 0) { temp_lenght = 1;}

CNormal[x] /= temp_lenght;
CNormal[y] /= temp_lenght;
CNormal[z] /= temp_lenght;


*v1x=CNormal[x];
*v1y=CNormal[y];
*v1z=CNormal[z];
}
