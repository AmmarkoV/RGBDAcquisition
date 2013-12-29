#ifndef TOOLS_H_INCLUDED
#define TOOLS_H_INCLUDED



int checkOpenGLError(char * file , int  line);
char * loadFileToMem(char * filename,unsigned long * file_length);

float RGB2OGL(unsigned int colr);

#endif // TOOLS_H_INCLUDED
