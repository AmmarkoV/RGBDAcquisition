#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <GL/gl.h>

#include "tools.h"

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */


#include <sys/time.h>
#include <unistd.h>
#include <time.h>

unsigned long tickBase = 0;

unsigned long GetTickCountMilliseconds()
{
   //This returns a monotnic "uptime" value in milliseconds , it behaves like windows GetTickCount() but its not the same..
   struct timespec ts;
   if ( clock_gettime(CLOCK_MONOTONIC,&ts) != 0) { return 0; }

   if (tickBase==0)
   {
     tickBase = ts.tv_sec*1000 + ts.tv_nsec/1000000;
     return 0;
   }

   return ( ts.tv_sec*1000 + ts.tv_nsec/1000000 ) - tickBase;
}



void printOpenGLError(int errorCode)
{
  switch (errorCode)
  {
    case  GL_NO_ERROR       :
         fprintf(stderr,"No error has been recorded.");
        break;
    case  GL_INVALID_ENUM   :
         fprintf(stderr,"An unacceptable value is specified for an enumerated argument. The offending command is ignored and has no other side effect than to set the error flag.\n");
        break;
    case  GL_INVALID_VALUE  :
         fprintf(stderr,"A numeric argument is out of range. The offending command is ignored and has no other side effect than to set the error flag.");
        break;
    case  GL_INVALID_OPERATION :
         fprintf(stderr,"The specified operation is not allowed in the current state. The offending command is ignored and has no other side effect than to set the error flag.");
        break;
    //case  GL_INVALID_FRAMEBUFFER_OPERATION :
    //     fprintf(stderr,"The framebuffer object is not complete. The offending command is ignored and has no other side effect than to set the error flag.");
    //    break;
    case  GL_OUT_OF_MEMORY :
         fprintf(stderr,"There is not enough memory left to execute the command. The state of the GL is undefined, except for the state of the error flags, after this error is recorded.");
        break;
    case  GL_STACK_UNDERFLOW :
         fprintf(stderr,"An attempt has been made to perform an operation that would cause an internal stack to underflow.");
        break;
    case  GL_STACK_OVERFLOW :
         fprintf(stderr,"An attempt has been made to perform an operation that would cause an internal stack to overflow.");
     break;
  };
}

int checkOpenGLError(char * file , int  line)
{
  int err=glGetError();
  if (err !=  GL_NO_ERROR /*0*/ )
    {
      fprintf(stderr,RED "OpenGL Error (%d) : %s %d \n ", err , file ,line );
      printOpenGLError(err);
      fprintf(stderr,"\n" NORMAL);
      return 1;
    }
 return 0;
}

char * loadFileToMem(char * filename,unsigned long * file_length)
{
  if (filename==0)  { fprintf(stderr,"Could not load shader incorrect filename \n"); return 0; }
  if (file_length==0)  { fprintf(stderr,"Could not load shader %s , incorrect file length parameter \n",filename); return 0; }

  FILE * pFile;
  long lSize;
  char * buffer;
  size_t result;

  pFile = fopen ( filename , "rb" );
  if (pFile==0) { fprintf(stderr,"Could not open shader file %s \n",filename); return 0;}

  // obtain file size :
  fseek (pFile , 0 , SEEK_END);
  lSize = ftell (pFile);
  rewind (pFile);

  // allocate memory to contain the whole file:
  buffer = (char*) malloc ( (sizeof(char)*lSize)+1 );
  if (buffer == 0) {fprintf(stderr,"Could not allocate %u bytes of memory for shader file %s \n",(unsigned int ) lSize,filename);   fclose (pFile); return 0; }

  // copy the file into the buffer:
  result = fread (buffer,1,lSize,pFile);
  if (result != lSize) {fputs ("Reading error",stderr); free(buffer);  fclose (pFile); return 0; }

  /* the whole file is now loaded in the memory buffer. */

  // terminate
  fclose (pFile);

  buffer[lSize]=0; //Add a null termination for shader usage
  *file_length = lSize;
  return buffer;
}

float RGB2OGL(unsigned int colr)
{
  return (float) colr/255;
}
