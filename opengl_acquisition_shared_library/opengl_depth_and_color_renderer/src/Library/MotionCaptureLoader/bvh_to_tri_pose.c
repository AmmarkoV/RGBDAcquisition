#include <stdio.h>
#include <stdlib.h>
#include "bvh_to_tri_pose.h"

#include "../TrajectoryParser/InputParser_C.h"

unsigned int bvh_resolveBVHToTRIJoint(struct bvhToTRI * bvhtri,const char * jName)
{
  unsigned int jID=0;

  for (jID=0; jID<bvhtri->numberOfJointAssociations; jID++)
  {
    if (strcmp(jName,bvhtri->jointAssociation[jID].bvhJointName)==0)
    {
      return jID;
    }
  }
 return 0;
}



int bvh_loadBVHToTRI(const char * filename , struct bvhToTRI * bvhtri)
{
  FILE * fp = fopen(filename,"r");
  if (fp!=0)
  {
   struct InputParserC * ipc = InputParser_Create(4096,7);
   InputParser_SetDelimeter(ipc,0,' ');
   InputParser_SetDelimeter(ipc,1,'(');
   InputParser_SetDelimeter(ipc,2,',');
   InputParser_SetDelimeter(ipc,3,')');
   InputParser_SetDelimeter(ipc,4,'\t');
   InputParser_SetDelimeter(ipc,5,10);
   InputParser_SetDelimeter(ipc,6,13);


    ssize_t read;
    char * line = NULL;
    size_t len = 0;

    char nameA[512]={0};

    char rotA[512]={0};
    char rotB[512]={0};
    char rotC[512]={0};

    float offsetA=0.0;
    float offsetB=0.0;
    float offsetC=0.0;

    bvhtri->numberOfJointAssociations=0;
    unsigned int jID=0;

    while  ((read = getline(&line, &len, fp)) != -1)
    {
       int num = InputParser_SeperateWords(ipc,line,1);

       if (num>0)
       {
        if (InputParser_WordCompareAuto(ipc,0,"JOINT_ASSOCIATION"))
         {
           jID=bvhtri->numberOfJointAssociations;
           InputParser_GetWord(ipc,1,bvhtri->jointAssociation[jID].bvhJointName,MAX_BVH_JOINT_NAME);
           InputParser_GetWord(ipc,2,bvhtri->jointAssociation[jID].triJointName,MAX_BVH_JOINT_NAME);
           //--------------------------------------------------
           fprintf(
                   stderr,"Associated #%u `%s` => `%s` \n",
                   jID,
                   bvhtri->jointAssociation[jID].bvhJointName,
                   bvhtri->jointAssociation[jID].triJointName
                  );

           bvhtri->jointAssociation[jID].useJoint=1;
           ++bvhtri->numberOfJointAssociations;
         } else
        if (InputParser_WordCompareAuto(ipc,0,"JOINT_ROTATION_ORDER"))
         {
           InputParser_GetWord(ipc,1,nameA,512);
           jID = bvh_resolveBVHToTRIJoint(bvhtri,nameA);
           InputParser_GetWord(ipc,2,bvhtri->jointAssociation[jID].rotationOrder[0].label,64);
           InputParser_GetWord(ipc,3,bvhtri->jointAssociation[jID].rotationOrder[1].label,64);
           InputParser_GetWord(ipc,4,bvhtri->jointAssociation[jID].rotationOrder[2].label,64);

           fprintf(stderr,"RotationOrder %s(%s,%s,%s)\n",nameA,
                     bvhtri->jointAssociation[jID].rotationOrder[0].label,
                     bvhtri->jointAssociation[jID].rotationOrder[1].label,
                     bvhtri->jointAssociation[jID].rotationOrder[2].label);
         } else
        if (InputParser_WordCompareAuto(ipc,0,"JOINT_OFFSET"))
        {
          InputParser_GetWord(ipc,1,nameA,512);
          jID = bvh_resolveBVHToTRIJoint(bvhtri,nameA);
          bvhtri->jointAssociation[jID].offset[0] = InputParser_GetWordFloat(ipc,2);
          bvhtri->jointAssociation[jID].offset[1] = InputParser_GetWordFloat(ipc,3);
          bvhtri->jointAssociation[jID].offset[2] = InputParser_GetWordFloat(ipc,4);
          fprintf(stderr,"Offset %s #%u (%0.2f,%0.2f,%0.2f)\n",nameA,jID,offsetA,offsetB,offsetC);
        }
       }
    }

    fprintf( stderr,"%u Associations\n", bvhtri->numberOfJointAssociations );


    if (line) { free(line); }

    fclose(fp);
    InputParser_Destroy(ipc);
  }
return 0;
}













