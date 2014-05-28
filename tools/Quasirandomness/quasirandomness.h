#ifndef QUASIRANDOMNESS_H_INCLUDED
#define QUASIRANDOMNESS_H_INCLUDED


struct quasiRandomizerContext
{
  unsigned int width , height , depth;
  float m_CurrentPos_x  , m_CurrentPos_y ,m_CurrentPos_z;
  unsigned long m_Base2 , m_Base3 , m_Base5;



};

int initializeQuasirandomnessContext(struct quasiRandomizerContext * qrc,unsigned int width , unsigned int height , unsigned int depth,int randomSeed);
int getNextRandomPoint(struct quasiRandomizerContext * qrc , float * x, float * y , float * z);

#endif // QUASIRANDOMNESS_H_INCLUDED
