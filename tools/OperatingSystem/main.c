#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>

#include "OperatingSystem.h"


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
     if (directoryList[i]==',')
     {
       lastPathEnd=i;
       if (itemNum == pathsEncountered)
       {
           fprintf(stderr,"strncpy( from %u to %u ) \n",lastPathStart,lastPathEnd);
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

  struct dirent *file; // a 'directory entity' AKA file
  struct stat info; // info about the file.

  output[0]=0;
  int addedItems = 0;

  fprintf(stderr,"Completely unsafe , no bounds check listDirectory call done now\n");
  while (file = readdir(dh))
   {
    if ( (strcmp(file->d_name,".")==0) || (strcmp(file->d_name,"..")==0) )
    {
     //Current or Parent directory..!
    }    else
    {
     //Regular Folder
     stat(file->d_name, &info);
     if (S_ISDIR(info.st_mode ))
         {
          if (addedItems!=0) { strcat(output,","); }
          strcat(output,file->d_name);
          ++addedItems;
         }
    }
    //printf("note: info.st_mode => %i\n", info.st_mode);
    //if (S_ISREG(info.st_mode)) printf("REGULAR FILE FOUND! %s\n", file->d_name);
   }
  closedir(dh);

  return 1;
}
