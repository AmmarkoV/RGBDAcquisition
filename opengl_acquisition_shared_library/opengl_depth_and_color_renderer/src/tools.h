/** @file tools.h
 *  @brief  Some pretty basic tools for OGL rendering
 *  @author Ammar Qammaz (AmmarkoV)
 */


#ifndef TOOLS_H_INCLUDED
#define TOOLS_H_INCLUDED




/**
* @brief check for the last opengl error occured
* @ingroup OGLTools
* @param String describing the file where the checkOpenGLError is called ( __FILE__ )
* @param Integer describing the line where the checkOpenGLError is called ( __LINE__ )
* @retval 1=Error , 0=No Error
*/
int checkOpenGLError(char * file , int  line);

/**
* @brief Load a file to memory buffer ( returned by this call )
* @ingroup OGLTools
* @param String of filename of the file to load
* @param Output Integer that will say how big the memory chunk loaded is
* @retval 0=Error , A pointer to the block of memory with contents from filename
*/
char * loadFileToMem(char * filename,unsigned long * file_length);

/**
* @brief Cast an unsigned char to a float to represent intensity of a color channel 0-255 to 0-1.0f
* @ingroup OGLTools
* @param Color in unsigned int 0-255
* @retval A float describing the color intensity
*/
float RGB2OGL(unsigned int colr);


/**
* @brief Calculate the distance between two 3d points
* @ingroup OGLTools
* @param Point 1 - X
* @param Point 1 - Y
* @param Point 1 - Z
* @param Point 2 - X
* @param Point 2 - Y
* @param Point 2 - Z
* @retval A float describing the distance
*/
float calculateDistance(float from_x,float from_y,float from_z,float to_x,float to_y,float to_z);


int rayIntersectsTriangle(float *p, float *d,float *v0, float *v1, float *v2);
int rayIntersectsRectangle(float *p, float *d,float *v0, float *v1, float *v2, float *v3);


double distanceBetween3DPoints(double * p1, double * p2);
float distanceBetween3DPointsFast(float *x1,float*y1,float *z1,float *x2,float*y2,float *z2);
float squaredDistanceBetween3DPoints(float *x1,float*y1,float *z1,float *x2,float*y2,float *z2);

void vectorDirection(float src_x,float src_y,float src_z,float targ_x,float targ_y,float targ_z,float *vect_x,float *vect_y,float *vect_z);
void findNormal(float *v1x, float *v1y, float *v1z, float v2x, float v2y, float v2z, float v3x, float v3y, float v3z );

#endif // TOOLS_H_INCLUDED
