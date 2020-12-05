#include "bvh_to_c.h"
#include "../edit/bvh_rename.h"
#include <stdio.h> 


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */

void bvh_print_C_Header(struct BVH_MotionCapture * bvhMotion)
{
  fprintf(stderr,RED "bvh_print_C_Header needs a complete rewrite..\n" NORMAL);
  fprintf(stdout,"/**\n");
  fprintf(stdout," * @brief An array with BVH string labels\n");
  fprintf(stdout," */\n");
  fprintf(stdout,"static const char * BVHOutputArrayNames[] =\n");
  fprintf(stdout,"{\n");
  char comma=',';
  char coord;//='X'; This is overwritten so dont need to be assigned..
  unsigned int countOfChannels=0;
  for (unsigned int i=0; i<bvhMotion->jointHierarchySize; i++)
  {
    if (bvhMotion->jointHierarchy[i].isRoot)
    {
    if (bvhMotion->jointHierarchy[i].hasPositionalChannels)
        {
           coord='X'; fprintf(stdout,"\"%s_%cposition\"%c // 0\n",bvhMotion->jointHierarchy[i].jointName,coord,comma);
           coord='Y'; fprintf(stdout,"\"%s_%cposition\"%c // 1\n",bvhMotion->jointHierarchy[i].jointName,coord,comma);
           coord='Z'; fprintf(stdout,"\"%s_%cposition\"%c // 2\n",bvhMotion->jointHierarchy[i].jointName,coord,comma);
           countOfChannels+=3;
        }

    if (bvhMotion->jointHierarchy[i].hasRotationalChannels)
        {
           if (bvhMotion->jointHierarchy[i].hasQuaternionRotation) 
           { 
            coord='W'; fprintf(stdout,"\"%s_%crotation\"%c // 3\n",bvhMotion->jointHierarchy[i].jointName,coord,comma);
            ++countOfChannels;
           }
           coord='Z'; fprintf(stdout,"\"%s_%crotation\"%c // 3\n",bvhMotion->jointHierarchy[i].jointName,coord,comma);
           coord='Y'; fprintf(stdout,"\"%s_%crotation\"%c // 4\n",bvhMotion->jointHierarchy[i].jointName,coord,comma);
           coord='X'; fprintf(stdout,"\"%s_%crotation\"%c // 5\n",bvhMotion->jointHierarchy[i].jointName,coord,comma);
           countOfChannels+=3;
        }
    } else
    {
     if (!bvhMotion->jointHierarchy[i].isEndSite)
        {
            for (unsigned int z=0; z<bvhMotion->jointHierarchy[i].loadedChannels; z++)
                {
                  ++countOfChannels;
                  if (countOfChannels+1>=bvhMotion->numberOfValuesPerFrame)
                  {
                      comma=' ';
                  }

                  unsigned int cT = bvhMotion->jointHierarchy[i].channelType[z];
                  fprintf(stdout,"\"%s_%s\"%c // %u\n ",bvhMotion->jointHierarchy[i].jointName,channelNames[cT],comma,countOfChannels);
                }

        }
    }
  }
  fprintf(stdout,"};\n\n\n\n");


  char label[513]={0};
  comma=',';
  //coord='X'; It is always reassigned before use..
  countOfChannels=0;

  fprintf(stdout,"/**\n");
  fprintf(stdout," * @brief This is a programmer friendly enumerator of joint output extracted from the BVH file.\n");
  fprintf(stdout," */\n");
  fprintf(stdout,"enum BVH_Output_Joints\n");
  fprintf(stdout,"{\n");
  for (unsigned int i=0; i<bvhMotion->jointHierarchySize; i++)
  {
    if (i==0)
        {
           coord='X'; snprintf(label,512,"%s_%cposition",bvhMotion->jointHierarchy[i].jointName,coord);
           uppercase(label);
           fprintf(stdout,"BVH_MOTION_%s = 0,\n",label);

           coord='Y'; snprintf(label,512,"%s_%cposition",bvhMotion->jointHierarchy[i].jointName,coord);
           uppercase(label);
           fprintf(stdout,"BVH_MOTION_%s,//1 \n",label);

           coord='Z'; snprintf(label,512,"%s_%cposition",bvhMotion->jointHierarchy[i].jointName,coord);
           uppercase(label);
           fprintf(stdout,"BVH_MOTION_%s,//2 \n",label);

           if (bvhMotion->jointHierarchy[i].hasQuaternionRotation) 
           { 
            coord='W'; snprintf(label,512,"%s_%crotation",bvhMotion->jointHierarchy[i].jointName,coord);
            uppercase(label);
            fprintf(stdout,"BVH_MOTION_%s,//3 \n",label);
           }

           coord='Z'; snprintf(label,512,"%s_%crotation",bvhMotion->jointHierarchy[i].jointName,coord);
           uppercase(label);
           fprintf(stdout,"BVH_MOTION_%s,//3 \n",label);

           coord='Y'; snprintf(label,512,"%s_%crotation",bvhMotion->jointHierarchy[i].jointName,coord);
           uppercase(label);
           fprintf(stdout,"BVH_MOTION_%s,//4 \n",label);

           coord='X'; snprintf(label,512,"%s_%crotation",bvhMotion->jointHierarchy[i].jointName,coord);
           uppercase(label);
           fprintf(stdout,"BVH_MOTION_%s,//5 \n",label);

           countOfChannels+=5;
        } else
        {
         if (!bvhMotion->jointHierarchy[i].isEndSite)
          {
            for (unsigned int z=0; z<bvhMotion->jointHierarchy[i].loadedChannels; z++)
                {
                  ++countOfChannels;
                  if (countOfChannels+1>=bvhMotion->numberOfValuesPerFrame)
                  {
                      comma=' ';
                  }

                  unsigned int cT = bvhMotion->jointHierarchy[i].channelType[z];
                  snprintf(label,512,"%s_%s",bvhMotion->jointHierarchy[i].jointName,channelNames[cT]);
                  uppercase(label);
                  fprintf(stdout,"BVH_MOTION_%s%c//%u \n",label,comma,countOfChannels);
                }
          }
        }
  }
  fprintf(stdout,"};\n\n\n");



  comma=',';
  //coord='X'; It is always reassigned before use..
  countOfChannels=0;

  fprintf(stdout,"/**\n");
  fprintf(stdout," * @brief This is a programmer friendly enumerator to access 3D output  extracted from the BVH file.\n");
  fprintf(stdout," *  Use ./GroundTruthDumper --from dataset/headerWithHeadAndOneMotion.bvh --printc  to extract this automatically */ \n");
  fprintf(stdout,"enum BVH_3D_Output_Joints\n");
  fprintf(stdout,"{\n");
  for (unsigned int i=0; i<bvhMotion->jointHierarchySize; i++)
  {
     snprintf(label,512,"%s",bvhMotion->jointHierarchy[i].jointName);
     uppercase(label);
     coord='X';
     fprintf(stdout,"BVH_3DPOINT_%s%c%c//%u \n",label,coord,comma,countOfChannels);
     ++countOfChannels;
     coord='Y';
     fprintf(stdout,"BVH_3DPOINT_%s%c%c//%u \n",label,coord,comma,countOfChannels);
     ++countOfChannels;
     coord='Z';
     fprintf(stdout,"BVH_3DPOINT_%s%c%c//%u \n",label,coord,comma,countOfChannels);
     ++countOfChannels;
  }
  fprintf(stdout,"};\n\n\n");



  comma=',';
  //coord='X'; It is always reassigned before use..
  countOfChannels=0;

  fprintf(stdout,"/**\n");
  fprintf(stdout," * @brief This is a programmer friendly enumerator to access 3D output  extracted from the BVH file.\n");
  fprintf(stdout," *  Use ./GroundTruthDumper --from dataset/headerWithHeadAndOneMotion.bvh --printc  to extract this automatically\n");
  fprintf(stdout," */\n");
  fprintf(stdout,"enum BVH_2D_Output_Joints\n");
  fprintf(stdout,"{\n");
  for (unsigned int i=0; i<bvhMotion->jointHierarchySize; i++)
  {
     snprintf(label,512,"%s",bvhMotion->jointHierarchy[i].jointName);
     uppercase(label);
     coord='X';
     fprintf(stdout,"BVH_2DPOINT_%s%c%c//%u \n",label,coord,comma,countOfChannels);
     ++countOfChannels;
     coord='Y';
     fprintf(stdout,"BVH_2DPOINT_%s%c%c//%u \n",label,coord,comma,countOfChannels);
     ++countOfChannels;
  }
  fprintf(stdout,"};\n\n\n");


  comma=',';
  //coord='X'; It is always reassigned before use..
  countOfChannels=0;

  fprintf(stdout,"/**\n");
  fprintf(stdout," * @brief This is a programmer friendly enumerator to access 3D output  extracted from the BVH file.\n");
  fprintf(stdout," *  Use ./GroundTruthDumper --from dataset/headerWithHeadAndOneMotion.bvh --printc  to extract this automatically\n");
  fprintf(stdout," */\n");
  fprintf(stdout,"enum BVH_2D_Output_Joints\n");
  fprintf(stdout,"{\n");
  for (unsigned int i=0; i<bvhMotion->jointHierarchySize; i++)
  {
     snprintf(label,512,"%s",bvhMotion->jointHierarchy[i].jointName);
     uppercase(label); 
     fprintf(stdout,"BVH_JOINT_%s%c//%u \n",label,comma,countOfChannels);
     ++countOfChannels; 
  }
  fprintf(stdout,"};\n\n\n");

  fprintf(stdout,"/**\n");
  fprintf(stdout," * @brief An array with BVH string labels\n");
  fprintf(stdout," */\n");
  fprintf(stdout,"static const char * BVH3DPositionalOutputArrayNames[] =\n");
  fprintf(stdout,"{\n");
  comma=',';
  countOfChannels=0;
  for (unsigned int i=0; i<bvhMotion->jointHierarchySize; i++)
  {
           coord='X'; fprintf(stdout,"\"%s_%cposition\"%c // %u\n",bvhMotion->jointHierarchy[i].jointName,coord,comma,countOfChannels);
           ++countOfChannels;
           coord='Y'; fprintf(stdout,"\"%s_%cposition\"%c // %u\n",bvhMotion->jointHierarchy[i].jointName,coord,comma,countOfChannels);
           ++countOfChannels;
           coord='Z'; fprintf(stdout,"\"%s_%cposition\"%c // %u\n",bvhMotion->jointHierarchy[i].jointName,coord,comma,countOfChannels);
           ++countOfChannels;
  }
  fprintf(stdout,"};\n\n\n\n");


}
