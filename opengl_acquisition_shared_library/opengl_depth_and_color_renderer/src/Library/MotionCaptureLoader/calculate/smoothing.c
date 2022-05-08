#include "smoothing.h"

struct ButterWorthArray * filterArrayAtomic = 0;

int butterWorth_allocateAtomic(int numberOfSensors,float fsampling,float fcutoff)
{
  filterArrayAtomic = butterWorth_allocate(numberOfSensors,fsampling,fcutoff);
  return (filterArrayAtomic!=0);
}


int butterWorth_deallocateAtomic()
{
   butterWorth_deallocate(filterArrayAtomic);
   return 1;
}


float butterWorth_filterAtomic(int value,float unfilteredValue)
{
   if (filterArrayAtomic==0) { return unfilteredValue; }
   return butterWorth_filter(&filterArrayAtomic->sensors[value],unfilteredValue);
}
