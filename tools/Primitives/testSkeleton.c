#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "skeleton.h"
#include "jsonCocoSkeleton.h"
//#include "nao_geometry.h"

float visualizationScale = 3.0;
int frames =0;
int delay=20;

unsigned int broadCastOnly=0;


void broadcastHTTPSkeleton(struct skeletonHuman * skeletonFound,const char * whereto)
{
  char cmd[4096]={0};
  char part[1024]={0};
  unsigned int i=0;

  for (i=0; i<HUMAN_SKELETON_PARTS; i++)
  {
       snprintf(part,1024,"%0.2f_%0.2f_%0.2f_",
                skeletonFound->joint[i].x ,
                skeletonFound->joint[i].y ,
                skeletonFound->joint[i].z );
       strcat(cmd,part);
  }

  fprintf(stderr,"Will execute : \n");
  fprintf(stderr,"%s%s \n",whereto,cmd);

  char runStrCmd[4096]={0};
  snprintf(runStrCmd,4096,"wget -qO- \"%s%s\" &> /dev/null &",whereto,cmd);
  i=system(runStrCmd);

  if (i==0) { fprintf(stderr,"Success\n"); } else
            { fprintf(stderr,"Error\n");   }
}





int doSkeletonConversions( struct skeletonHuman * skel )
{
   char filenameBuf[512]={0};

   updateSkeletonAngles(skel);
   printSkeletonHuman(skel);



   if (
        (frames==0) || // In order to refresh the visualization frame 0 is needed
        (broadCastOnly==0) ||
        (broadCastOnly==frames)
       )
   {
   //updateSkeletonAnglesNAO(skel);
   fprintf(stderr,"doSkeletonConversions #%u ",frames);
   snprintf(filenameBuf,512,"skel2D%u.svg",frames);
   if (!skeleton2DEmpty(skel))
   {
    visualize2DSkeletonHuman(filenameBuf,  skel, visualizationScale);
   } else { fprintf(stderr,RED "Won't print out 2D scenes with skeletons for empty 2D skeleton info \n" NORMAL );}



   if (!skeleton3DEmpty(skel))
   {
   snprintf(filenameBuf,512,"skel3D.scene");
   visualize3DSkeletonHuman(filenameBuf,skel,frames);
   } else { fprintf(stderr,RED "Won't print out 3D scenes with skeletons for empty 3D skeleton info \n" NORMAL );}


   if (!skeleton3DEmpty(skel))
   {
    broadcastHTTPSkeleton(skel,"http://127.0.0.1:8080/sk.html?sk=");
   }else { fprintf(stderr,RED "Won't broadcast 3D scenes for empty 3D skeleton info\n" NORMAL );}
   }



//   struct naoCommand nao={0};
   struct skeletonHuman sk={0};

   //setNAOMotorsFromHumanSkeleton( &nao , &sk );
   snprintf(filenameBuf,512,"skel%u.skel",frames);
   //printoutNAOCommand( filenameBuf , &nao );


   usleep(delay*1000);
   cleanSkeleton(skel);
   ++frames;
  return 1;
}





int printJointField ( struct skeletonHuman * skel )
{
  unsigned int i=0;

  printf("numberOfJoints: %u\n",HUMAN_SKELETON_PARTS);



  printf("joints2D: [");
  for (i=0; i<HUMAN_SKELETON_PARTS-1; i++)
  {
     printf("%0.4f,%0.4f,",skel->joint2D[i].x,skel->joint2D[i].y);
  }
  printf("%0.4f,%0.4f]\n",skel->joint2D[HUMAN_SKELETON_PARTS-1].x,skel->joint2D[HUMAN_SKELETON_PARTS-1].y);


  printf("joints3D: [");
  for (i=0; i<HUMAN_SKELETON_PARTS-1; i++)
  {
     printf("%0.4f,%0.4f,%0.4f,",skel->joint[i].x,skel->joint[i].y,skel->joint[i].z);
  }
  ++i;
  printf("%0.4f,%0.4f,%0.4f]\n",skel->joint[HUMAN_SKELETON_PARTS-1].x,skel->joint[HUMAN_SKELETON_PARTS-1].y,skel->joint[HUMAN_SKELETON_PARTS-1].z);




  printf("relativeJointAngle: [");
  for (i=0; i<HUMAN_SKELETON_PARTS-1; i++)
  {
     printf("%0.4f,%0.4f,%0.4f,",skel->relativeJointAngle[i].x,skel->relativeJointAngle[i].y,skel->relativeJointAngle[i].z);
  }
  ++i;
  printf("%0.4f,%0.4f,%0.4f]\n",skel->relativeJointAngle[HUMAN_SKELETON_PARTS-1].x,skel->relativeJointAngle[HUMAN_SKELETON_PARTS-1].y,skel->relativeJointAngle[HUMAN_SKELETON_PARTS-1].z);




  printf("timestamp: %u\n", skel->observationNumber);
  printf("---\n");
  return 1;
}


int parseJointField ( struct skeletonHuman * skel , char * var , char * val)
{
  if (strcmp(var,"numberOfJoints")==0) { if ( atoi(val) != HUMAN_SKELETON_PARTS ) { fprintf(stderr,"Incorrect number of joints \n"); } } else
  if (strcmp(var,"timestamp")==0)      { skel->observationNumber= atoi(val); } else
  if (strcmp(var,"joints3D")==0)
         {
           char * numStart = strchr(val , '[');
           char * numEnd = strchr(val , ',');
           if ( (numStart!=0) && (numEnd!=0) )
           {
            ++numStart; // Supposing we have [xyz.etc,
            *numEnd=0;

            unsigned int i=0;
            for (i=0; i<HUMAN_SKELETON_PARTS; i++)
            {
              //Grab Coordinate X
              skel->joint[i].x=atof(numStart);
              numStart=numEnd+1;
              numEnd = strchr(numStart , ',');
              if (numEnd!=0) { *numEnd=0; }

              //Grab Coordinate Y
              skel->joint[i].y=atof(numStart);
              numStart=numEnd+1;
              if ( i==HUMAN_SKELETON_PARTS-1 )  { numEnd = strchr(numStart , ']'); } else
                                                { numEnd = strchr(numStart , ','); }
              if (numEnd!=0) { *numEnd=0; }

              //Grab Coordinate Z
              skel->joint[i].z=atof(numStart);
              numStart=numEnd+1;
              numEnd = strchr(numStart , ',');
              if (numEnd!=0) { *numEnd=0; }
            }
           }
         } else
  if (strcmp(var,"joints2D")==0)
         {
           char * numStart = strchr(val , '[');
           char * numEnd = strchr(val , ',');
           if ( (numStart!=0) && (numEnd!=0) )
           {
            ++numStart; // Supposing we have [xyz.etc,
            *numEnd=0;

            unsigned int i=0;
            for (i=0; i<HUMAN_SKELETON_PARTS; i++)
            {
              //Grab Coordinate Y
              skel->joint2D[i].x=atof(numStart);
              numStart=numEnd+1;
              if ( i==HUMAN_SKELETON_PARTS-1 )  { numEnd = strchr(numStart , ']'); } else
                                                { numEnd = strchr(numStart , ','); }
              if (numEnd!=0) { *numEnd=0; }

              //Grab Coordinate Z
              skel->joint2D[i].y=atof(numStart);
              numStart=numEnd+1;
              numEnd = strchr(numStart , ',');
              if (numEnd!=0) { *numEnd=0; }
            }
           }
         } else
         {
           return 0;
         }


 return 1;
}


int parseJointList(const char * filename)
{
  struct skeletonHuman skel={0};

  char * line = NULL;
  size_t len = 0;
  ssize_t read;

  FILE * fp = fopen(filename,"r");
  if (fp!=0)
  {

    char * line = NULL;
    size_t len = 0;

    while ((read = getline(&line, &len, fp)) != -1)
    {
       printf("Retrieved line of length %zu :\n", read);
       printf("%s", line);



      if (strstr(line,"---")!=0)
      {
           doSkeletonConversions( &skel );
           printJointField ( &skel );
      } else
      {
        char * varNameEnd = strchr(line , ':');
        if (varNameEnd!=0)
        {
         *varNameEnd=0;
         printf("VAR = %s\n", line);
         char * val = varNameEnd+1;
         printf("VAL = %s\n", val);

         parseJointField ( &skel , line ,  val  );
        }
       }
    }

    fclose(fp);
    if (line) { free(line); }
    return 1;
  }



  return 0;
}




int main(int argc, char *argv[])
{
    if (argc < 2 ) { fprintf(stderr,"Please give filename of joint list \n"); return 1; }

  fprintf(stderr,"Running Converter on %s !\n",argv[1]);


  unsigned int i=0;
  for (i=0; i<argc; i++)
  {
    if (strcmp(argv[i],"-broadcastOnly")==0) {
                                              broadCastOnly=atoi(argv[i+1]);
                                              fprintf(stderr,"Will Broadcast only skeleton %u\n",broadCastOnly);
                                             } else
    if (strcmp(argv[i],"-delay")==0) {
                                              delay=atoi(argv[i+1]);
                                              fprintf(stderr,"Delay set to %u\n",delay);
                                             }

  }




    if (strstr(argv[1],".json")!=0)
    {
      struct skeletonCOCO skel={0};
      parseJsonCOCOSkeleton(argv[1],&skel);

      unsigned int frameNumber = 0;
      if (argc>=3) { frameNumber=atoi(argv[2]); }
      printCOCOSkeletonCSV(&skel,frameNumber);
    } else
    {
     struct skeletonHuman defaultPose={0};
     fillWithDefaultSkeleton(&defaultPose);
     visualize2DSkeletonHuman("defaultPose.svg", &defaultPose , visualizationScale );

     struct skeletonHuman defaultNAOPose={0};
     fillWithDefaultNAOSkeleton(&defaultNAOPose);
     visualize2DSkeletonHuman("defaultNAOPose.svg", &defaultNAOPose , visualizationScale );

     parseJointList(argv[1]);
    }
    return 0;
}
