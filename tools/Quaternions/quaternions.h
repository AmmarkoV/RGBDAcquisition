/** @file quaternions.h
 *  @brief This is a small math library that deals with quaternions , doubles are used for maximum precision
 *
 *  @author Ammar Qammaz (AmmarkoV)
 */


#ifndef QUATERNIONS_H_INCLUDED
#define QUATERNIONS_H_INCLUDED


/**
 * @brief This enumerator defines the order of quaternions qWqXqYqZ means W X Y Z , and qXqYqZqW means X Y Z W
 */
enum quatOrder
{
  qWqXqYqZ=0,
  qXqYqZqW
};


/**
 * @brief This function converts euler angles to quaternions
 * @ingroup quaternions
 * @param Quaternions , The output quaternions in the order declared by enum quatOrder
 * @param Euler angles , The input euler angles X,Y,Z
 * @param The convention used for quaternions ( see enum quatOrder )
 * @retval Nothing , no return value
 */
void euler2Quaternions(double * quaternions,double * euler,int quaternionConvention);


/**
 * @brief This function converts quaternions to euler angles
 * @ingroup quaternions
 * @param Euler angles , The output euler angles X,Y,Z
 * @param Quaternions , The input quaternions in the order declared by enum quatOrder
 * @param The convention used for quaternions ( see enum quatOrder )
 * @retval Nothing , no return value
 */
void quaternions2Euler(double * euler,double * quaternions,int quaternionConvention);


int normalizeQuaternions(double *qX,double *qY,double *qZ,double *qW);

double innerProductQuaternions(double qAX,double qAY,double qAZ,double qAW ,
                               double qBX,double qBY,double qBZ,double qBW);

double anglesBetweenQuaternions(double qAX,double qAY,double qAZ,double qAW ,
                                double qBX,double qBY,double qBZ,double qBW);
#endif // QUATERNIONS_H_INCLUDED
