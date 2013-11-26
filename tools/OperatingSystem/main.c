#include <stdio.h>
#include <sys/types.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <dirent.h>

#include "OperatingSystem.h"

int listDirectory(char * directory , char * output, unsigned int maxOutput)
{
  DIR *dh = opendir(directory); // directory handle
  struct dirent *file; // a 'directory entity' AKA file
  struct stat info; // info about the file.

  output[0]=0;

  fprintf(stderr,"Completely unsafe , no bounds check listDirectory call done now\n");
  while (file = readdir(dh))
   {
    stat(file->d_name, &info);
    printf("note: file->d_name => %s\n", file->d_name);

    strcat(output,file->d_name);
    strcat(output,"\n");

    //printf("note: info.st_mode => %i\n", info.st_mode);
    //if (S_ISREG(info.st_mode)) printf("REGULAR FILE FOUND! %s\n", file->d_name);
   }
  closedir(dh);

  return 0;
}
