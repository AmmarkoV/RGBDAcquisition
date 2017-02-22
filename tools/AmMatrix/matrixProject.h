/** @file matrixCalculations.h
 *  @brief  Functions to prepare matrixes and transform 3D points
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef MATRIXPROJECT_H_INCLUDED
#define MATRIXPROJECT_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif


 
int _glhProjectf(float objx, float objy, float objz, float *modelview, float *projection, int *viewport, float *windowCoordinate);
int _glhUnProjectf(float winx, float winy, float winz, float *modelview, float *projection, int *viewport, float *objectCoordinate);



#ifdef __cplusplus
}
#endif


#endif // MATRIXPROJECT_H_INCLUDED
