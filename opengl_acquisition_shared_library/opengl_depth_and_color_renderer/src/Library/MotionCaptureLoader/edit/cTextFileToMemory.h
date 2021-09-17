/** @file cTextFileToMemory.h
 *  @brief  A header only library that can help parsing text files 
 *  https://github.com/AmmarkoV/CTextFileToMemory
 *  @author Ammar Qammaz (AmmarkoV)
 */
 
#ifndef BVH_INVERSEKINEMATICS_H_INCLUDED
#define BVH_INVERSEKINEMATICS_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#define C_TEXTFILETOMEMORY_VERSION 0.01


struct cTextFileToMemory
{
  char * ptr;
  char * buffer;
  unsigned long   bufferSize;
  
  unsigned int populatedStrings;
  unsigned int allocatedStrings;
  unsigned long   numberOfLines;
  
  char ** strings;
};


static unsigned char * ctftm_readFileToMemory(const char * filename,unsigned long *length)
{
    *length = 0;
    FILE * pFile = fopen ( filename , "rb" );

    if (pFile==0) { return 0; }

    // obtain file size:
    fseek (pFile , 0 , SEEK_END);
    unsigned long lSize = ftell (pFile);
    rewind (pFile);

    // allocate memory to contain the whole file:
    unsigned long bufferSize = sizeof(char)*(lSize+1);
    char * buffer = (char*) malloc (bufferSize);
    if (buffer == 0 ) { errno = ENOMEM; fclose(pFile); return 0; }

    // copy the file into the buffer:
    size_t result = fread (buffer,1,lSize,pFile);
    if (result != lSize)
    {
        free(buffer);
        fclose(pFile);
        return 0;
    }

    /* the whole file is now loaded in the memory buffer. */

    // terminate
    fclose (pFile);

    buffer[lSize]=0; //Null Terminate Buffer!
    *length = (unsigned int) lSize;
    return buffer;
}


static int ctftm_countNumberOfLines(struct cTextFileToMemory * ctftm)
{
    char * ptr = ctftm->buffer;
    char * limit = ptr + ctftm->bufferSize;
    
    ctftm->numberOfLines=0;
    
    while (ptr<limit)
    {
        switch (*ptr)
        {
          case 0   :                            break;
          case 10  :  ctftm->numberOfLines+=1;  break;
          case 13  :  ctftm->numberOfLines+=1;  break;  
          case EOF :                            break;
        };
        
        ++ptr; 
    }
  return ctftm->numberOfLines;
}




static unsigned int ctftm_parselines(struct cTextFileToMemory * ctftm)
{ 
    if (ctftm == NULL) 
    {
        errno = EINVAL;
        return -1;
    }
     
    char done = 0;
    char * ptrStart = ctftm->ptr;
    
    char *previousString = ptrStart;
    char * limit = ctftm->buffer + ctftm->bufferSize;
    
    
    //fprintf(stderr," ctftm_parselines Allocated strings %u ...\n",ctftm->allocatedStrings);
    //fprintf(stderr," ctftm_parselines Populated strings %u ...\n",ctftm->populatedStrings);
    if  ( (ctftm->allocatedStrings!=0) && (ctftm->populatedStrings==0) )
    {
     while (ctftm->ptr < limit)
     {
        switch (*ctftm->ptr)
        {
          case 0   :                  break;
          case 10  :
          case 13  :
          case EOF :
           *ctftm->ptr=0; 
            ctftm->strings[ctftm->populatedStrings]=previousString;
            ctftm->populatedStrings+=1; 
            previousString=ctftm->ptr+1;
          break;
        };
      ++ctftm->ptr;
     }
    } else
    {
        fprintf(stderr,"ctftm_parselines: No scanning done..\n");
    }
    
    
  return ctftm->ptr - ptrStart;
}



static int ctftm_loadTextFileToMemory(struct cTextFileToMemory * ctftm, const char * filename)
{
    if (ctftm==0)    { return 0; }
    if (filename==0) { return 0; }
    //----------------------------
    
    ctftm->buffer = ctftm_readFileToMemory(filename,&ctftm->bufferSize);
    ctftm->ptr    = ctftm->buffer;


    if (ctftm->buffer!=0)
    {
        ctftm_countNumberOfLines(ctftm);
        //fprintf(stderr," Number of lines in file %lu ...\n",ctftm->numberOfLines);
        
        ctftm->strings = (char **) malloc(ctftm->numberOfLines * sizeof(char **)); 
        ctftm->allocatedStrings = ctftm->numberOfLines;
        ctftm->populatedStrings = 0;
        //fprintf(stderr," ctftm_loadTextFileToMemory Allocated strings %u ...\n",ctftm->allocatedStrings);
        //fprintf(stderr," ctftm_loadTextFileToMemory Populated strings %u ...\n",ctftm->populatedStrings);
        
        if (ctftm->strings!=0)
        {
         ssize_t read = ctftm_parselines(ctftm);
         return (read>0);
        }
    }
    
    return 0;
}



static unsigned int ctftm_getNumberOfRecords(struct cTextFileToMemory * ctftm)
{
   return ctftm->numberOfLines; 
}

static char * ctftm_getRecords(struct cTextFileToMemory * ctftm,unsigned int record)
{
   return ctftm->strings[record]; 
}

#ifdef __cplusplus
}
#endif

#endif
