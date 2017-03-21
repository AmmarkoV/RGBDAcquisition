/** @file main.c
 *  @brief  A minimal binary that renders scene files using OGLRendererSandbox s
 *          X86 compilation: gcc -o -L/usr/X11/lib   main main.c -lGL -lX11 -lpng -ljpeg
 *          X64 compilation: gcc -o -L/usr/X11/lib64 main main.c -lGL -lX11 -lpng -ljpeg
 *  @author Ammar Qammaz (AmmarkoV)
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>


int processCommand(struct InputParserC * ipc , unsigned int label , char * line , unsigned int words_count ,HandBodyModels::UpperBodyModel &converter, std::vector<HandBodyModels::UpperBodyModel> &outputModels)
{
  if (InputParser_WordCompareAuto(ipc,0,"FRAME"))
   {
    //fprintf(stderr,"Found Frame %u \n",InputParser_GetWordInt(ipc,1));
    if (InputParser_GetWordInt(ipc,1)!=0)
    {//Flush next frame
      outputModels.push_back(converter);
      fprintf(stdout,".");
      return 1;
    }
   }


 unsigned int i=0;
 for (i=0; i<COCO_PARTS; i++)
 {
  if (InputParser_WordCompareAuto(ipc,1,(char*) COCOBodyNames[i]))
   {
    if (BJ::INCORRECT != cocoMapToUpperBodyTrackerBodyJoint[i])
    {
     converter.setPosition(
                           (BJ::BodyJointType) cocoMapToUpperBodyTrackerBodyJoint[i],
                            cv::Point3f(
                                        InputParser_GetWordFloat(ipc,3),
                                        InputParser_GetWordFloat(ipc,4),
                                        InputParser_GetWordFloat(ipc,5)
                                       )
                          );
      //fprintf(stdout,"Joint %s = ( %0.2f %0.2f %0.2f )\n" , COCOBodyNames[i] , InputParser_GetWordFloat(ipc,3), InputParser_GetWordFloat(ipc,4), InputParser_GetWordFloat(ipc,5));
      return 1;
    }
   }
 }
  return 0;
}
int main(int argc, char **argv)
{


    return 0;
}



int main(int argc, char** argv)
{
   std::vector<HandBodyModels::UpperBodyModel> outputModels;
   HandBodyModels::UpperBodyModel converter;

   char filename[]="coco.scene";
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
             processCommand(ipc,0,line,words_count,converter,outputModels);
         } // End of line containing tokens
    } //End of getting a line while reading the file
  }

  fclose(fp);
  InputParser_Destroy(ipc);

  fprintf(stdout,"\nDumping to CSV\n");
  HandBodyModels::UpperBodyModel::saveToCsvFile("test.csv",outputModels);
}
