#ifndef QUATERNIONS_H_INCLUDED
#define QUATERNIONS_H_INCLUDED


enum quatOrder
{
  qWqXqYqZ=0,
  qXqYqZqW
};

void euler2Quaternions(double * quaternions,double * euler,int quaternionConvention);
void quaternions2Euler(double * quaternions,double * euler,int quaternionConvention);



#endif // QUATERNIONS_H_INCLUDED
