
#include "smoothing.h"

struct ButterWorthArray * filterArrayAtomic = 0;

int butterWorth_allocateAtomic(int numberOfSensors,float fsampling,float fcutoff)
{
  //printf("butterWorth_allocateAtomic(%u,%0.2f,%0.2f)\n",numberOfSensors,fsampling,fcutoff);
  filterArrayAtomic = butterWorth_allocate(numberOfSensors,fsampling,fcutoff);
  return (filterArrayAtomic!=0);
}


int butterWorth_deallocateAtomic()
{
  //printf("butterWorth_deallocateAtomic()\n");
  butterWorth_deallocate(filterArrayAtomic);
  return 1;
}


float butterWorth_filterAtomic(int value,float unfilteredValue)
{
  //printf("butterWorth_filterAtomic(%u,%0.2f)\n",value,unfilteredValue);
  if (filterArrayAtomic==0) { return unfilteredValue; }
  if (value>=filterArrayAtomic->numberOfSensors) { return unfilteredValue; }
  return butterWorth_filter(&filterArrayAtomic->sensors[value],unfilteredValue);
}
