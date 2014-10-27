#ifndef PATTERNS_H_INCLUDED
#define PATTERNS_H_INCLUDED

struct pattern
{
  unsigned int maxStates;
  unsigned int currentStates;
  unsigned int * state;
  unsigned int * duration;
};



#endif // PATTERNS_H_INCLUDED
