#include "patterns.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ABSDIFF(num1,num2) ( (num1-num2) >=0 ? (num1-num2) : (num2 - num1) )

int viewPattern(struct pattern* pat , const char * label )
{
  unsigned int i=0;

  fprintf(stderr,"Pattern %s is : ",label);
  for (i=0; i<pat->currentStates; i++)
  {
   fprintf(stderr,"%u(%u) ",pat->state[i],pat->duration[i]);
  }
  fprintf(stderr,"\n");

 return 1;
}

int convertStringToPattern(struct pattern * out , const char *  in)
{
  if (out->state!=0)    { free(out->state); }
  if (out->duration!=0) { free(out->duration); }

  out->currentStates = strlen(in);
  out->maxStates = out->currentStates;
  out->state = (unsigned int * ) malloc( out->currentStates * sizeof (unsigned int ));
  out->duration = (unsigned int * ) malloc( out->currentStates * sizeof (unsigned int ));

  int i=0;
  for (i=0; i<out->currentStates; i++)
  {
    out->state[i]=in[i];
    out->duration[i]=1;
  }
 return 1;
}


int compactPattern(struct pattern * observation)
{
  unsigned int i=0,offset=0,initialStates=observation->currentStates;

 for (i=0; i+offset+1<initialStates; i++)
  {
    if (offset!=0)
     {
      observation->state[i+1] = observation->state[i+1+offset];
     }

    if ( observation->state[i] == observation->state[i+1] )
      {
         observation->duration[i]+=observation->duration[i+1];
         ++offset;
         if (i>0) { --i; }
      }
  }
  observation->currentStates=i+1;
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

  //STEP 3 : RE COMPACT
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
     } else
     {
        fprintf(stderr,"Mismatch at character %u/%u ( %u )  vs %u/%u  ( %u )\n",
                rP,remembered->currentStates,remembered->state[rP] ,
                oP,observed->currentStates,observed->state[oP]);
       return 0;
     }
    //---------------------------------------------------------
     if (
         (oP==observed->currentStates) ||
         (rP==remembered->currentStates)
        )
        {
          if (remembered->currentStates>observed->currentStates)
          {
            fprintf(stderr,"Mismatch size , our sequence is good up to a point but not complete\n");
            return 0;
          }

          done=1;
        }
  }

  return 1;
}





