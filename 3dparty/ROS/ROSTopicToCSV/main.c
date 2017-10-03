#include <stdio.h>
#include <stdlib.h>
#include "../../../tools/Primitives/CompareBody/InputParser_C.h"


int parseFile(const char * filename)
{
 printf("Parsing rostopic %s !\n",filename);
 //fprintf(stderr,"Running COCO 2D skeleton (%s)\n",filename);

//  char * line = NULL;
//  size_t len = 0;
  ssize_t read;
  unsigned int frameNumber =0;

  FILE * fp = fopen(filename,"r");
  if (fp!=0)
  {
   struct InputParserC * ipc = InputParser_Create(1024,4);

   InputParser_SetDelimeter(ipc,0,':');
   InputParser_SetDelimeter(ipc,1,'[');
   InputParser_SetDelimeter(ipc,2,',');
   InputParser_SetDelimeter(ipc,3,']');

    char type[512];
    char subtype[512];
    char content[512];
    char str[512];
    char * line = NULL;
    size_t len = 0;


    while ((read = getline(&line, &len, fp)) != -1)
    {
       //printf("Retrieved line of length %zu :\n", read);
      printf("%s", line);
      if (strstr(line,"---")!=0)
      {
       printf("NEXT MESSAGE , do flush here\n");
      } else
      {
      char * foundColonNL=strstr(line,": \n");
      if (foundColonNL!=0)
      {
       *foundColonNL=0;
       snprintf(type,512,"%s",line);
       printf("PREVIOUSLINETYPE(%s)\n",type);
      } else
      {
        char * foundContentAfterColon=strstr(line,": ");
        if (
             (line[0]=' ') &&
             (line[1]=' ') &&
             (foundContentAfterColon!=0)
            )
        {
           *foundContentAfterColon=0;
           snprintf(subtype,512,"%s",line);
           snprintf(content,512,"%s",foundContentAfterColon+1);
           printf("SUBTYPE(%s)\n",subtype);
           printf("CONTENT(%s)\n",content);
        }

           if (strstr(line,"joints")!=0)
      {
       printf("%s", line);
       int i=0;
       int num = InputParser_SeperateWords(ipc,line,1);




      ++frameNumber;
       }
      }
      }
      }
     if (line) { free(line); }

    fclose(fp);
    return 1;
  }
  fprintf(stderr,"Unable to open (%s)\n",filename);
  return 0;
}



int main(int argc, char** argv)
{
    if (argc<=1) { fprintf(stderr,"No arguments provided..\n"); return 1; }
    parseFile(argv[1]);
    return 0;
}
