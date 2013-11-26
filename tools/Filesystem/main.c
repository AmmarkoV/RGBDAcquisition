#include <stdio.h>
#include <sys/types.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <dirent.h>

#include "FilesystemListing.h"



int listDirectory(char * directory , char * output, unsigned int maxOutput)
{
  DIR *dh = opendir(directory); // directory handle
  struct dirent *file; // a 'directory entity' AKA file
  struct stat info; // info about the file.


  while (file = readdir(dh))
   {
    stat(file->d_name, &info);
    printf("note: file->d_name => %s\n", file->d_name);
    //printf("note: info.st_mode => %i\n", info.st_mode);
    //if (S_ISREG(info.st_mode)) printf("REGULAR FILE FOUND! %s\n", file->d_name);
   }
  closedir(dh);

  return 0;
}
