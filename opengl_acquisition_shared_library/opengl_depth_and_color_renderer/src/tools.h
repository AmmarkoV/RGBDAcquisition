#ifndef TOOLS_H_INCLUDED
#define TOOLS_H_INCLUDED



int checkOpenGLError(char * file , int  line);
char * loadFileToMem(char * filename,unsigned long * file_length);

float RGB2OGL(unsigned int colr);


float calculateDistance(float from_x,float from_y,float from_z,float to_x,float to_y,float to_z);

#endif // TOOLS_H_INCLUDED
