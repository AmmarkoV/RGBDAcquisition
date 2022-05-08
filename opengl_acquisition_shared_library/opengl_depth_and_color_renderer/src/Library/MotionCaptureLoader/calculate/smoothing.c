
#include "smoothing.h"
 
void * butterWorth_allocateAtomic(int numberOfSensors,float fsampling,float fcutoff)
{
  //printf("butterWorth_allocateAtomic(%u,%0.2f,%0.2f)\n",numberOfSensors,fsampling,fcutoff);
  struct ButterWorthArray * filterArrayAtomic = butterWorth_allocate(numberOfSensors,fsampling,fcutoff);
  return (void *) filterArrayAtomic;
}


int butterWorth_deallocateAtomic(void * handle)
{
  struct ButterWorthArray * filterArrayAtomic = (struct ButterWorthArray *) handle;
  butterWorth_deallocate(filterArrayAtomic);
  return 1;
}


float butterWorth_filterAtomic(void * handle,int value,float unfilteredValue)
{
  struct ButterWorthArray * filterArrayAtomic = (struct ButterWorthArray *) handle;
  if (filterArrayAtomic==0) { return unfilteredValue; }
  if (value>=filterArrayAtomic->numberOfSensors) { return unfilteredValue; }
  return butterWorth_filter(&filterArrayAtomic->sensors[value],unfilteredValue);
}
