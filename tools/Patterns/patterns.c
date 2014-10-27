#include "patterns.h"
#include <stdio.h>
#include <stdlib.h>

#define ABSDIFF(num1,num2) ( (num1-num2) >=0 ? (num1-num2) : (num2 - num1) )


int compactPattern(struct pattern * observation)
{
  unsigned int i=0,offset=0;
  for (i=1; i+offset<observation->currentStates; i++)
  {
    if ( observation->state[i-1] == observation->state[i+offset] )
      {
         observation->duration[i-1]+=observation->duration[i+offset];
         ++offset;
      }
  }
  observation->currentStates -= offset;
  return 1;
}



int cleanPattern(struct pattern * observation , double noiseFactor)
{
  //STEP 1 : COMPACT
  compactPattern(observation);

  //STEP 2 : DENOISE
  unsigned int i=0 , offset=0;
  for (i=2; i<observation->currentStates; i++)
  {
    if ( observation->state[i-2] == observation->state[i] )
      {
        if ( (observation->duration[i-2] + observation->duration[i]) * noiseFactor > observation->duration[i-1] )
            {
              fprintf(stderr,"Eliminated state %u , keeping its %u duration \n",i-1,observation->duration[i-1]);
              observation->state[i-1]=observation->state[i];
              ++offset;
            }
      }
  }


  if (offset!=0)
  { //We made some changes so let's compact again..!
    compactPattern(observation);
  }

 return 1;
}






int patternsMatch(struct pattern * remembered, struct pattern * observed)
{
  unsigned int done=0 , rP =0 , oP = 0 , timingMismatch = 0;

  while (!done)
  {
     if (remembered->state[rP] == observed->state[oP])
     {
       timingMismatch+=ABSDIFF(remembered->duration[rP],observed->duration[oP]);
       ++rP; ++oP;
     }




     if (oP==observed->currentStates) { done=1; }
  }


  for (oP=0; oP<observed->currentStates; oP++)
  {



    //Let's Check




  }





  return 1;
}





