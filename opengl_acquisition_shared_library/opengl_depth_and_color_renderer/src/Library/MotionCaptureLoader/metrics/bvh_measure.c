#include "bvh_measure.h"

#include "../ik/hardcodedProblems_inverseKinematics.h"
#include "../edit/bvh_rename.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>



#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */




void compareMotionBuffers(const char * msg,struct MotionBuffer * guess,struct MotionBuffer * groundTruth)
{
  fprintf(stderr,"%s \n",msg);
  fprintf(stderr,"___________\n");

  if (guess->bufferSize != groundTruth->bufferSize)
  {
    fprintf(stderr,"compareMotionBuffers: Buffer Size mismatch..\n");
    return ;
  }

  //--------------------------------------------------
  fprintf(stderr,"Guess : ");
  for (unsigned int i=0; i<guess->bufferSize; i++)
  {
    fprintf(stderr,"%0.2f " ,guess->motion[i]);
  }
  fprintf(stderr,"\n");
  //--------------------------------------------------
  fprintf(stderr,"Truth : ");
  for (unsigned int i=0; i<groundTruth->bufferSize; i++)
  {
    fprintf(stderr,"%0.2f " ,groundTruth->motion[i]);
  }
  fprintf(stderr,"\n");
  //--------------------------------------------------


  fprintf(stderr,"Diff : ");

  for (unsigned int i=0; i<guess->bufferSize; i++)
  {
    float diff=fabs(groundTruth->motion[i] - guess->motion[i]);
    if (fabs(diff)<0.1) { fprintf(stderr,GREEN "%0.2f " ,diff); } else
                         { fprintf(stderr,RED "%0.2f " ,diff); }
  }
  fprintf(stderr,NORMAL "\n___________\n");
}


void compareTwoMotionBuffers(struct BVH_MotionCapture * mc,const char * msg,struct MotionBuffer * guessA,struct MotionBuffer * guessB,struct MotionBuffer * groundTruth)
{
  fprintf(stderr,"%s \n",msg);
  fprintf(stderr,"___________\n");

  if ( (guessA->bufferSize != groundTruth->bufferSize) || (guessB->bufferSize != groundTruth->bufferSize) )
  {
    fprintf(stderr,"compareTwoMotionBuffers: Buffer Size mismatch..\n");
    return ;
  }


  fprintf(stderr,"Diff : ");
  for (unsigned int i=0; i<guessA->bufferSize; i++)
  {
    float diffA=fabs(groundTruth->motion[i] - guessA->motion[i]);
    float diffB=fabs(groundTruth->motion[i] - guessB->motion[i]);
    if ( (diffA==0.0) && (diffA==diffB) )  { fprintf(stderr,BLUE  "%0.2f ",diffA-diffB); } else
    {
     if (diffA>=diffB)                     { fprintf(stderr,GREEN "%0.2f ",diffA-diffB); } else
                                           { fprintf(stderr,RED   "%0.2f ",diffB-diffA); }

     unsigned int jID =mc->motionToJointLookup[i].jointID;
     unsigned int chID=mc->motionToJointLookup[i].channelID;
     fprintf(stderr,NORMAL "(%s#%u/%0.2f->%0.2f/%0.2f) ",mc->jointHierarchy[jID].jointName,chID,guessA->motion[i],guessB->motion[i],groundTruth->motion[i]);
    }
  }
  fprintf(stderr,NORMAL "\n___________\n");
}





int bvhMeasureIterationInfluence(
    struct BVH_MotionCapture * mc,
    float lr,
    float spring,
    unsigned int iterations,
    unsigned int epochs,
    unsigned int fIDPrevious,
    unsigned int fIDSource,
    unsigned int fIDTarget,
    unsigned int multiThreaded
)
{
    int result=0;

    struct BVH_Transform bvhTargetTransform= {0};

    const int MAX_ITERATIONS=25;
                   
    struct simpleRenderer renderer= {0};
    const float distance=-150;
    simpleRendererDefaults( &renderer, 1920, 1080, 582.18394,   582.52915 );// https://gopro.com/help/articles/Question_Answer/HERO4-Field-of-View-FOV-Information
    simpleRendererInitialize(&renderer);

    fprintf(stderr,"BVH file has motion files with %u elements\n",mc->numberOfValuesPerFrame);

    float initialMAEInPixels=0.0,finalMAEInPixels=0.0;
    float initialMAEInMM=0.0,finalMAEInMM=0.0;

    //Load all motion buffers
    struct MotionBuffer * groundTruth         = mallocNewMotionBuffer(mc);
    struct MotionBuffer * initialSolution     = mallocNewMotionBuffer(mc);
    struct MotionBuffer * solution            = mallocNewMotionBuffer(mc);
    struct MotionBuffer * previousSolution    = mallocNewMotionBuffer(mc);
    struct MotionBuffer * penultimateSolution = mallocNewMotionBuffer(mc);



    if ( (solution!=0) && (groundTruth!=0) && (initialSolution!=0) && (previousSolution!=0) )
    {
        
      for (int startFrame=60; startFrame<mc->numberOfFrames; startFrame+=30)
      for (int testIteration=0; testIteration<4; testIteration++)
      {
        fIDTarget = startFrame;
        if (testIteration==0) { fIDSource=fIDTarget-5;  } else
        if (testIteration==1) { fIDSource=fIDTarget-40; } else
        if (testIteration==2) { fIDSource=fIDTarget-60; } else
        if (testIteration==3) { fIDSource=fIDTarget-80; } 
        fprintf(stderr,"Started test iteration testing Source %u / Previous %u \n",fIDSource,fIDPrevious);
        
        
        if (
            ( bvh_copyMotionFrameToMotionBuffer(mc,initialSolution,fIDSource) ) &&
            ( bvh_copyMotionFrameToMotionBuffer(mc,previousSolution,fIDPrevious) ) &&
            ( bvh_copyMotionFrameToMotionBuffer(mc,solution,fIDSource) ) &&
            ( bvh_copyMotionFrameToMotionBuffer(mc,groundTruth,fIDTarget) )
           )
        {
            //------------------------------------------------------------------------
            if(fIDPrevious>0)
            {
              bvh_copyMotionFrameToMotionBuffer(mc,penultimateSolution,fIDPrevious-1);
            } else
            {
              bvh_copyMotionFrameToMotionBuffer(mc,penultimateSolution,fIDPrevious); 
            }
            //------------------------------------------------------------------------
            initialSolution->motion[0]=0;
            initialSolution->motion[1]=0;
            initialSolution->motion[2]=distance;
            //-----------------------------------------
            previousSolution->motion[0]=0;
            previousSolution->motion[1]=0;
            previousSolution->motion[2]=distance;
            //-----------------------------------------
            penultimateSolution->motion[0]=0;
            penultimateSolution->motion[1]=0;
            penultimateSolution->motion[2]=distance;
            //-----------------------------------------
            solution->motion[0]=0;
            solution->motion[1]=0;
            solution->motion[2]=distance;
            //-----------------------------------------
            groundTruth->motion[0]=0;
            groundTruth->motion[1]=0;
            groundTruth->motion[2]=distance;
            //------------------------------------------------------------------------


            if ( bvh_loadTransformForMotionBuffer(mc,groundTruth->motion,&bvhTargetTransform,0) )
            {
                if  (bvh_projectTo2D(mc,&bvhTargetTransform,&renderer,0,0))
                {
                    struct ikConfiguration ikConfig= {0};
                    //------------------------------------
                    ikConfig.learningRate = lr;
                    ikConfig.iterations = iterations;
                    ikConfig.epochs = epochs;
                    ikConfig.spring = spring;
                    ikConfig.gradientExplosionThreshold = 50;
                    ikConfig.dumpScreenshots = 0;
                    ikConfig.maximumAcceptableStartingLoss=0.0; // Dont use this
                    ikConfig.verbose = 0;
                    ikConfig.tryMaintainingLocalOptima=1; //Less Jittery but can be stuck at local optima
                    ikConfig.dontUseSolutionHistory=1; //<- should I use solution history ?
                    ikConfig.ikVersion = IK_VERSION;
                    //------------------------------------


    
                      struct ikProblem * problem = (struct ikProblem * ) malloc(sizeof(struct ikProblem));
                      if (problem!=0)
                                     { memset(problem,0,sizeof(struct ikProblem)); } else
                                     { fprintf(stderr,RED "Failed to allocate memory for our IK problem..\n" NORMAL);  return 0; }

                      if (!prepareDefaultBodyProblem(
                                                       problem,
                                                       mc,
                                                       &renderer,
                                                       previousSolution,
                                                       solution,
                                                       &bvhTargetTransform
                                                      )
                         )
                         {
                               fprintf(stderr,RED "Could not prepare the problem for IK solution\n" NORMAL);
                               free(problem);
                               return 0;
                         }
                   
                   
                   
                   
                   //fprintf(stdout,"Iterations,MAE,FPS\n"); 
                   for (ikConfig.iterations=1; ikConfig.iterations<MAX_ITERATIONS; ikConfig.iterations++)
                   {
                     //--------------------------------------------------------
                     bvh_copyMotionFrameToMotionBuffer(mc,solution,fIDSource);
                     solution->motion[0]=0;
                     solution->motion[1]=0;
                     solution->motion[2]=distance;
                     //--------------------------------------------------------

                     
                   unsigned long startTime = GetTickCountMicrosecondsIK();

                    if (
                        approximateBodyFromMotionBufferUsingInverseKinematics(
                            mc,
                            &renderer,
                            problem,
                            &ikConfig,
                            //---------------
                            penultimateSolution,  
                            previousSolution,
                            solution,
                            groundTruth,
                            //-------------------
                            &bvhTargetTransform,
                            //-------------------
                            multiThreaded, //Use a single thread unless --mt is supplied before testik command!
                            //------------------- 
                            &initialMAEInPixels,
                            &finalMAEInPixels,
                            &initialMAEInMM,
                            &finalMAEInMM
                        )
                    )
                    {
                        //Important :)
                        //=======
                        result=1;
                        //=======
                        unsigned long endTime = GetTickCountMicrosecondsIK();


                        //-------------------------------------------------------------------------------------------------------------
                        //compareMotionBuffers("The problem we want to solve compared to the initial state",initialSolution,groundTruth);
                        //compareMotionBuffers("The solution we proposed compared to ground truth",solution,groundTruth);
                        //-------------------------------------------------------------------------------------------------------------
                        compareTwoMotionBuffers(mc,"Improvement",initialSolution,solution,groundTruth);

                        //fprintf(stderr,"MAE in 2D Pixels went from %0.2f to %0.2f \n",initialMAEInPixels,finalMAEInPixels);
                        fprintf(stderr,"MAE in 3D mm went from %0.2f to %0.2f \n",initialMAEInMM*10,finalMAEInMM*10);
                        fprintf(stderr,"Computation time was %lu microseconds ( %0.2f fps )\n",endTime-startTime,convertStartEndTimeFromMicrosecondsToFPSIK(startTime,endTime));
                        
                        
                        float fpsResult =convertStartEndTimeFromMicrosecondsToFPSIK(startTime,endTime);
                        float accResult=finalMAEInMM*10;
                        fprintf(stderr,"test %u | %u->%u / %u | %u/%u iteration / %0.2f MM / %0.2f \n",testIteration,fIDSource,fIDTarget,mc->numberOfFrames,ikConfig.iterations,MAX_ITERATIONS,accResult,fpsResult);
                        
                        // .dat output
                        fprintf(stdout,"%u %0.2f %0.2f %u\n",ikConfig.iterations,accResult,fpsResult,testIteration); 
                        
                    }
                    else
                    {
                        fprintf(stderr,RED "Failed to run IK code..\n" NORMAL);
                    }
                   }
                   
                   
                   
                   
                  //Cleanup allocations needed for the problem..
                  cleanProblem(problem);
                  free(problem); 
                  problem=0;
                }
                else
                {
                    fprintf(stderr,"Could not project 2D points of target..\n");
                }
            }
        }
        
       fprintf(stdout,"\n\n"); 
      } //Test loop
        freeMotionBuffer(&previousSolution);
        previousSolution = 0; // Double make sure that variable is clean
        freeMotionBuffer(&solution);
        solution = 0;         // Double make sure that variable is clean
        freeMotionBuffer(&initialSolution);
        initialSolution = 0;  // Double make sure that variable is clean
        freeMotionBuffer(&groundTruth);
        groundTruth = 0;      // Double make sure that variable is clean
    } //Have correct data
          
    return result;
}





int extractMinimaMaximaFromBVHList(const char * filename)
{ //This will go in bvh_measure.c

 FILE * fp = fopen(filename,"r");
    if (fp!=0)
        {
           
            struct BVH_MotionCapture bvhMotion={0}; 
            unsigned int numberOfValues=0; 
            
            
            float * minima = 0;
            float * maxima = 0;
            unsigned int minmaxSize = 0;
            
            char * line = NULL;
            size_t len = 0;
            ssize_t read;
 
            unsigned int fileNumber=0;
            while  ( (read = getline(&line, &len, fp)) != -1) 
                {
                  if (line!=0)
                  {
                    int lineLength = strlen(line);
                    if (lineLength>=1)
                    {
                      if (line[lineLength-1]==10) { line[lineLength-1]=0; }
                      if (line[lineLength-1]==13) { line[lineLength-1]=0; }
                    }
                    if (lineLength>=2)
                    {
                      if (line[lineLength-2]==10) { line[lineLength-2]=0; }
                      if (line[lineLength-2]==13) { line[lineLength-2]=0; }
                    }

                  if (bvhMotion.motionValues!=0)   
                    {
                      //Instead of regular freeing we do this weird free to avoid freeing last structure to able to access its mIDs 
                      bvh_free(&bvhMotion);
                      fprintf(stderr,"Freed file `%s`\n",line); 
                    }
                    
                  fprintf(stderr,"Next file is `%s`\n",line);
                  if ( bvh_loadBVH(line, &bvhMotion, 1.0) )
                   {
                      fprintf(stderr,"Loaded file `%s`\n",line);
                      //Change joint names..
                      bvh_renameJointsForCompatibility(&bvhMotion);
                      fprintf(stderr,"Did rename `%s`\n",line);
                      
                      if (numberOfValues!=0)
                      {
                          if (numberOfValues != bvhMotion.numberOfValuesPerFrame)
                          {
                              fprintf(stderr,"Incorrect number of arguments..!");
                              fprintf(stderr,"Expected %u and got %u..!",numberOfValues,bvhMotion.numberOfValuesPerFrame);
                              exit(1);
                          }
                      }
                      
                      numberOfValues = bvhMotion.numberOfValuesPerFrame;
                      
                      fprintf(stderr,"There needs to be a check for alternating BVH file sizes..!");
                      if (minima==0)
                           { 
                             minima = (float*) malloc(sizeof(float) * bvhMotion.numberOfValuesPerFrame); 
                             minmaxSize=bvhMotion.numberOfValuesPerFrame; 
                             memset(minima,0,sizeof(float) * bvhMotion.numberOfValuesPerFrame);
                           }
                      if (maxima==0)
                           { 
                             maxima = (float*) malloc(sizeof(float) * bvhMotion.numberOfValuesPerFrame); 
                             minmaxSize=bvhMotion.numberOfValuesPerFrame; 
                             memset(maxima,0,sizeof(float) * bvhMotion.numberOfValuesPerFrame);
                           }
            
                      if ( (minima!=0) && (maxima!=0) )
                        {
                            if (minmaxSize!=bvhMotion.numberOfValuesPerFrame)
                            {
                               fprintf(stderr,"Inconsistent BVH files of different number of parameters given.. Terminating.. \n");            
                               free(minima); minima=0;
                               free(maxima); maxima=0;
                               bvh_free(&bvhMotion);
                               fclose(fp);
                               return 0; 
                            }
                        }
                      
                      
                      unsigned int mIDAbsolute=0;
                      for (unsigned int fID=0; fID<bvhMotion.numberOfFrames; fID++)
                      {
                        for (unsigned int mID=0; mID<bvhMotion.numberOfValuesPerFrame; mID++)
                         {
                            if (bvhMotion.motionValues[mIDAbsolute]<minima[mID]) { minima[mID]=bvhMotion.motionValues[mIDAbsolute]; }
                            if (bvhMotion.motionValues[mIDAbsolute]>maxima[mID]) { maxima[mID]=bvhMotion.motionValues[mIDAbsolute]; }
                            ++mIDAbsolute; 
                         }
                      }

                      //bvh_free(&bvhMotion);
                      //fprintf(stderr,"Freed file `%s`\n",line);
                   }
                  }

                  ++fileNumber;
                  //if (fileNumber==10) { break; }
                }
          
           if ( (maxima!=0) && (minima!=0) ) 
                          {
                             fprintf(stdout,"\n\n\n//Minima/Maxima for %u files :\n\n",fileNumber);
                             fprintf(stdout,"float minimumLimits[%u]={0};\n",numberOfValues);
                             fprintf(stdout,"float maximumLimits[%u]={0};\n",numberOfValues);
                             fprintf(stdout,"//--------------------------\n");
                             for (unsigned int mID=7; mID<numberOfValues; mID++)
                                            {
                                               unsigned int jID = bvhMotion.motionToJointLookup[mID].jointID;
                                               if (minima[mID]!=0.0) { fprintf(stdout,"minimumLimits[%u]=%0.2f;//jID=%u -> %s\n",mID,minima[mID],jID,bvhMotion.jointHierarchy[jID].jointName); }  
                                               if (maxima[mID]!=0.0) { fprintf(stdout,"maximumLimits[%u]=%0.2f;//jID=%u -> %s\n",mID,maxima[mID],jID,bvhMotion.jointHierarchy[jID].jointName); }
                                            }
                             fprintf(stdout,"\n\n//--------------------------\n");
                           } else
                           {
                               fprintf(stdout,"Error with number of values, possibly mixed files..\n");
                           }
          
          bvh_free(&bvhMotion);
          
          if (line!=0) { 
                         fprintf(stderr,"Freed final file `%s`\n",line);
                         free(line); 
                       }
                       
            //Done using memory
            if(minima!=0) { free(minima); minima=0; }
            if(maxima!=0) { free(maxima); maxima=0; }
          
          fclose(fp);
          return 1;
        }
  return 0;
}
