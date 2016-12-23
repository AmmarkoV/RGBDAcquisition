#ifndef XWDLIB_H_INCLUDED
#define XWDLIB_H_INCLUDED

int initXwdLib(int argc, char  **argv);
int closeXwdLib();
int getScreen(unsigned char * frame , unsigned int * frameWidth , unsigned int * frameHeight);

#endif // XWDLIB_H_INCLUDED
