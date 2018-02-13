#include "groundTruthParser.h"
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <vector>

#include "InputParser_C.h"

const char * groundTruthFilename=0;
unsigned int groundTruthModuleID=0;
unsigned int groundTruthDevID=0;

std::vector<std::vector<float> >  groundTruthParams;

int groundTruthProviderStart(unsigned int moduleID , unsigned int devID , const char * directory)
{
   groundTruthFilename=directory;
   groundTruthDevID=devID;
   groundTruthModuleID=moduleID;
return 1;
}


const char * getGroundTruthPath()
{
 return groundTruthFilename;
}

int populateGroundTruth(const char * filename)
{
   #define LINE_MAX_LENGTH 1024
   unsigned int readOpResult = 0;
   unsigned int linesRead = 0;
   unsigned int i;
   char line [LINE_MAX_LENGTH]={0};
   char buf[512];

   groundTruthParams.clear();

   //Try and open filename
   FILE * fp = fopen(filename,"r");
   if (fp == 0 ) { fprintf(stderr,"Cannot open ground truth file %s \n",filename); return 0; }

   //Allocate a token parser
   struct InputParserC * ipc=0;
   ipc = InputParser_Create(LINE_MAX_LENGTH,5);
   InputParser_SetDelimeter(ipc,0,' ');
   InputParser_SetDelimeter(ipc,1,',');
   InputParser_SetDelimeter(ipc,2,';');

   if (ipc==0)  { fprintf(stderr,"Cannot allocate memory for new stream\n"); fclose(fp); return 0; }

   //Everything is set , Lets read the file!
   while (!feof(fp))
   {
    //We get a new line out of the file
    readOpResult = (fgets(line,LINE_MAX_LENGTH,fp)!=0);
    if ( readOpResult != 0 )
    {
      //We tokenize it
      unsigned int words_count = InputParser_SeperateWords(ipc,line,0);
      if ( words_count > 0 )
         {
           if (linesRead==0)
           {
             fprintf(stderr,"Joints Provided For Ground Truth ..!\n");
             for (i=0; i<words_count; i++)
             {
               InputParser_GetWord(ipc,i,buf,512);
             }
           } else
           {
             std::vector<float>  newGroundTruthLine;
             fprintf(stderr,"\nFrame %u , Words %u | " ,linesRead,words_count );
             for (i=0; i<words_count; i++)
             {
                float newVal = InputParser_GetWordFloat(ipc,i);
                fprintf(stderr,"%0.2f,", newVal);
                newGroundTruthLine.push_back(newVal);
             }
             groundTruthParams.push_back(newGroundTruthLine);
             fprintf(stderr,"\n");
           }

         } // End of line containing tokens
      ++linesRead;
    } //End of getting a line while reading the file
   }

  fclose(fp);
  InputParser_Destroy(ipc);

  return 1;
}

int groundTruthProviderSetupSkinnedModel(
                                    char * modelPath,
                                    double * minimumValues,
                                    double * varianceValues,
                                    double * maximumValues,
                                    unsigned int numberOfValues
                                   )
{

   char groudTruthFilenameFinalPath[2048];
   snprintf(groudTruthFilenameFinalPath , 2048 , "frames/%s/hyps.txt" , groundTruthFilename );
   return populateGroundTruth(groudTruthFilenameFinalPath);

}

int groundTruthProviderStop()
{
return 0;
}

float *  getGroundTruth(unsigned char * colorFrame , unsigned int colorWidth , unsigned int colorHeight ,
                        unsigned short * depthFrame  , unsigned int depthWidth , unsigned int depthHeight ,
                        struct calibration  * fc ,
                        unsigned int frameNumber ,
                        unsigned int *groundTruthLength
                        )
{
 if (frameNumber >= groundTruthParams.size())
    {
     fprintf(stdout,"FAILED TO Find groundTruthProviderNewFrame for frame %u ..\n",frameNumber);
     return 0;
    }

 std::vector<float>  what2Return = groundTruthParams[frameNumber];

 float * what2ReturnF = (float * ) malloc(sizeof(float) * what2Return.size());
 *groundTruthLength = what2Return.size();

 for(unsigned int i=0; i<what2Return.size(); i++)
 {
  what2ReturnF[i]=what2Return[i];
 }

 return what2ReturnF;
}



int groundTruthProviderNewFrame(
                                unsigned char * colorFrame , unsigned int colorWidth , unsigned int colorHeight ,
                                unsigned short * depthFrame  , unsigned int depthWidth , unsigned int depthHeight ,
                                struct calibration  * fc ,
                                unsigned int frameNumber
                               )
{
 unsigned int groundTruthLength;
 float *  what2Return = getGroundTruth(
                                       colorFrame,colorWidth,colorHeight,
                                       depthFrame,depthWidth,depthHeight,
                                       fc ,
                                       frameNumber ,
                                       &groundTruthLength
                                      );

 ///DO WHAT YOU WANT HERE WITH WHAT2RETURN VARIABLE


 if (what2Return!=0)
    {

     free(what2Return);


     return 1;
    }


 return 0;
}






int main(int argc, char *argv[])
{

  if (argc<2)
     { fprintf(stderr,"./GroundTruthParser filename.csv\n"); return 0; }


     populateGroundTruth(argv[1]);

  return 0;
}



