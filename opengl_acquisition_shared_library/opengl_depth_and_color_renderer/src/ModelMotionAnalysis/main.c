/** @file main.c
 *  @brief  A minimal binary that renders scene files using OGLRendererSandbox s
 *          X86 compilation: gcc -o -L/usr/X11/lib   main main.c -lGL -lX11 -lpng -ljpeg
 *          X64 compilation: gcc -o -L/usr/X11/lib64 main main.c -lGL -lX11 -lpng -ljpeg
 *  @author Ammar Qammaz (AmmarkoV)
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "../TrajectoryParser/InputParser_C.h"

int processCommand(struct InputParserC * ipc ,char * line , unsigned int words_count)
{
  if (InputParser_WordCompareAuto(ipc,0,"POSE"))
   {
    //fprintf(stderr,"Found Frame %u \n",InputParser_GetWordInt(ipc,1));
    char str[512];
    if (InputParser_GetWord(ipc,1,str,512)!=0)
    {//Flush next frame
      fprintf(stdout,".");
      return 1;
    }
   }


  return 0;
}
int main(int argc, char **argv)
{
 char filename[]="hyps.scene";
 char line [512]={0};

 fprintf(stdout,"Opening file %s\n",filename);
   FILE * fp = fopen(filename,"r");
   if (fp == 0 ) { fprintf(stderr,"Cannot open trajectory stream %s \n",filename); return 0; }

   struct InputParserC * ipc=0;
   ipc = InputParser_Create(512,5);
   if (ipc==0)  { fprintf(stderr,"Cannot allocate memory for new stream\n"); fclose(fp); return 0; }

   while (!feof(fp))
   {
   //We get a new line out of the file
   int readOpResult = (fgets(line,512,fp)!=0);
   if ( readOpResult != 0 )
    {
      //We tokenize it
      unsigned int words_count = InputParser_SeperateWords(ipc,line,0);
      if ( words_count > 0 )
         {
             processCommand(ipc,line,words_count);
         } // End of line containing tokens
    } //End of getting a line while reading the file
  }

  fclose(fp);
  InputParser_Destroy(ipc);

 return 0;
}
