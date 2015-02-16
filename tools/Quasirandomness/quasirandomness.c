#include "quasirandomness.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>


unsigned long  ulrandom()
{
   return ((unsigned long)rand() << 32) | (unsigned long)rand();
}

int initializeQuasirandomnessContext(struct quasiRandomizerContext * qrc,unsigned int width , unsigned int height , unsigned int depth,int randomSeed)
{
  if (qrc==0) { return 0; }
  srand(time(0));
  memset(qrc,0,sizeof(struct quasiRandomizerContext));
  qrc->width=width;
  qrc->height=height;
  qrc->depth=depth;
  if (randomSeed)
  {
   qrc->m_Base2=ulrandom();
   qrc->m_Base3=ulrandom();
   qrc->m_Base5=ulrandom();
  }

  return 1;
}

//Halton Quasirandom algorithm
int getNextRandomPoint(struct quasiRandomizerContext * qrc , float * x, float * y , float * z)
{
        float fOneOver3 = 1.0f/3.0f;
        float fOneOver5 = 1.0f/5.0f;

        long oldBase2 = qrc->m_Base2;
        qrc->m_Base2++;
        long diff = qrc->m_Base2 ^ oldBase2;

        float s = 0.5f;

        do
        {
            if ((oldBase2 & 1) == 1)
                qrc->m_CurrentPos_x -= s;
            else
                qrc->m_CurrentPos_x += s;

            s *= 0.5f;

            diff = diff >> 1;
            oldBase2 = oldBase2 >> 1;
        }
        while (diff > 0);

        long bitmask = 0x3;
        long bitadd  = 0x1;
        s = fOneOver3;

        qrc->m_Base3++;

        while (1)
        {
            if ((qrc->m_Base3 & bitmask) == bitmask)
            {
                qrc->m_Base3 += bitadd;
                qrc->m_CurrentPos_y -= 2 * s;

                bitmask = bitmask << 2;
                bitadd  = bitadd  << 2;

                s *= fOneOver3;
            }
            else
            {
                qrc->m_CurrentPos_y += s;
                break;
            }
        };
        bitmask = 0x7;
        bitadd  = 0x3;
        long dmax = 0x5;

        s = fOneOver5;

        qrc->m_Base5++;

        while (1)
        {
            if ((qrc->m_Base5 & bitmask) == dmax)
            {
                qrc->m_Base5 += bitadd;
                qrc->m_CurrentPos_z -= 4 * s;

                bitmask = bitmask << 3;
                dmax = dmax << 3;
                bitadd  = bitadd  << 3;

                s *= fOneOver5;
            }
            else
            {
                qrc->m_CurrentPos_z += s;
                break;
            }
        };

        *x=qrc->m_CurrentPos_x*qrc->width;
        *y=qrc->m_CurrentPos_y*qrc->height;
        *z=qrc->m_CurrentPos_z*qrc->depth;

        return qrc->m_Base2;
    }


