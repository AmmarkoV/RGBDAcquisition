/** @file main.c
*   @brief  Some Operating System specific functions to abstract the OS layer
*   @author Ammar Qammaz (AmmarkoV)
*   @bug This does not have implementations for non-Linux systems
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <signal.h>

#include "OperatingSystem.h"

#define MAX_SYSTEM_PATH 4096




int copyDirectoryListItem(int itemNum , char * directoryList , char * output, unsigned int maxOutput)
{
  unsigned int i=0;
  unsigned int directoryListLength = strlen(directoryList);


  unsigned int numberOfChars=0;
  unsigned int pathsEncountered =0;
  unsigned int lastPathStart =0;
  unsigned int lastPathEnd =0;
  char * lastPath = directoryList;

  while ( i < directoryListLength )
  {
     if ( (directoryList[i]==',') || (i+1 == directoryListLength) )
     {
       lastPathEnd=i;
       //If we finished the string there is an extra character!
       if (i+1 == directoryListLength) { ++lastPathEnd; }

       if (itemNum == pathsEncountered)
       {
           //fprintf(stderr,"strncpy( from %u to %u ) \n",lastPathStart,lastPathEnd);
           if (lastPathEnd<=lastPathStart)
            {
             fprintf(stderr,"Cannot Copy wrong bounds \n");
             return 0;
            }

           numberOfChars=lastPathEnd-lastPathStart;
           if (numberOfChars>=maxOutput)
            {
             fprintf(stderr,"Cannot Copy out of max bounds %u\n",maxOutput);
             return 0;
            }

           strncpy(output,lastPath,numberOfChars);
           output[numberOfChars]=0;
           return 1;
       }

       lastPathStart = i+1;
       lastPath=directoryList+i+1;
       ++pathsEncountered;
     }

    ++i;
  }

 return 0;
}


int listDirectory(char * directory , char * output, unsigned int maxOutput)
{
  DIR *dh = opendir(directory); // directory handle

  if (dh==0)
     {
        fprintf(stderr,"Could not list directory %s\n",directory);
        return 0;
     }

  char * fullPathToFilename = (char*) malloc ( (MAX_SYSTEM_PATH+1) * sizeof(char) );
  struct dirent *file; // a 'directory entity' AKA file
  struct stat info; // info about the file.

  //clear output string
  output[0]=0;
  unsigned int addedChars = 0;
  unsigned int addedItems = 0;
  unsigned int charsToBeAdded=0;

  fprintf(stderr,"Completely unsafe , no bounds check listDirectory call done now\n");
  while (file = readdir(dh))
   {
    if ( (strcmp(file->d_name,".")==0) || (strcmp(file->d_name,"..")==0) )
    {
     //Current or Parent directory..!
    }    else
    {
     //Regular Folder
     if (strlen(directory)+strlen(file->d_name)+3 < MAX_SYSTEM_PATH )
     {
      //Our Accomodation for the full string is enough
      sprintf(fullPathToFilename,"%s/%s",directory,file->d_name);

      if ( stat(fullPathToFilename, &info) == 0 ) /*SUCCESS*/
      {
       if ( (S_ISDIR(info.st_mode)) && (!S_ISREG(info.st_mode)) )
          {
           charsToBeAdded = strlen(file->d_name)+1;

           if (addedChars+charsToBeAdded<maxOutput)
           {
            if (addedItems!=0) { strcat(output,","); }
            strcat(output,file->d_name);
            ++addedItems;
            addedChars+=charsToBeAdded;
           }

          }
      } else
      { fprintf(stderr,"Error stating %s\n",file->d_name); }
     }
    }
   }

  closedir(dh);
  free(fullPathToFilename);

  fprintf(stderr,"listDirectory filled %u+1 chars of %u availiable\n",addedChars,maxOutput);

  return 1;
}



int registerTerminationSignal(void * callback)
{
  unsigned int failures=0;
  if (signal(SIGINT, callback)  == SIG_ERR)  { fprintf(stderr,"Acquisition client cannot handle SIGINT!\n"); ++failures; }
  if (signal(SIGHUP, callback)  == SIG_ERR)  { fprintf(stderr,"Acquisition client handle SIGHUP!\n");        ++failures; }
  if (signal(SIGTERM, callback) == SIG_ERR)  { fprintf(stderr,"Acquisition client handle SIGTERM!\n");       ++failures; }
  if (signal(SIGKILL, callback) == SIG_ERR)  { fprintf(stderr,"Acquisition client handle SIGKILL!\n");       ++failures; }

  return (failures==0);
}
