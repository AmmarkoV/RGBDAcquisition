#ifndef RECORDOUTPUT_H_INCLUDED
#define RECORDOUTPUT_H_INCLUDED


int logEvent(
              unsigned int frameNumber,
              float x,
              float y,
              float width,
              float height,
              const char * label,
              float probability
            );

#endif // RECORDOUTPUT_H_INCLUDED
