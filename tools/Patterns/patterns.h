#ifndef PATTERNS_H_INCLUDED
#define PATTERNS_H_INCLUDED

struct pattern
{
  unsigned int maxStates;
  unsigned int currentStates;
  unsigned int * state;
  unsigned int * duration;
};

int viewPattern(struct pattern* pat , const char * label );
int convertStringToPattern(struct pattern * out , const char *  in);

int cleanPattern(struct pattern * observation , double noiseFactor);
int patternsMatch(struct pattern * remembered, struct pattern * observed);

#endif // PATTERNS_H_INCLUDED
