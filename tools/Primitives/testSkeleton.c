#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "skeleton.h"

float visualizationScale = 3.0;
int frames =0;

int doSkeletonConversions( struct skeletonHuman * skel )
{
  //updateSkeletonAngles(skel);
  updateSkeletonAnglesNAO(skel);

  char filenameBuf[512]={0};
  snprintf(filenameBuf,512,"skel%u.svg",frames);

  visualizeSkeletonHuman(filenameBuf,  skel, visualizationScale);
  ++frames;
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
       // printf("Retrieved line of length %zu :\n", read);
       // printf("%s", line);
        char * varNameEnd = strchr(line , ':');
        if (varNameEnd!=0)
        {
         *varNameEnd=0;
         printf("VAR = %s\n", line);
         char * val = varNameEnd+1;
         printf("VAL = %s\n", val);

         parseJointField ( &skel , line ,  val );
           doSkeletonConversions( &skel );
         printJointField ( &skel );

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

    printf("Running Converter on %s !\n",argv[1]);

    struct skeletonHuman defaultPose={0};
    fillWithDefaultSkeleton(&defaultPose);
    visualizeSkeletonHuman("defaultPose.svg", &defaultPose , visualizationScale );

    struct skeletonHuman defaultNAOPose={0};
    fillWithDefaultNAOSkeleton(&defaultNAOPose);
    visualizeSkeletonHuman("defaultNAOPose.svg", &defaultNAOPose , visualizationScale );

    parseJointList(argv[1]);
    return 0;
}
