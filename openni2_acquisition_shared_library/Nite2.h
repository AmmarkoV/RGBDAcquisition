#ifndef NITE2_H_INCLUDED
#define NITE2_H_INCLUDED

int startNite2();
int stopNite2();
int loopNite2(unsigned int frameNumber);


int registerSkeletonDetectedEvent(void * callback);

#endif // NITE2_H_INCLUDED
