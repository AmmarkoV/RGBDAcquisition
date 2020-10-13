#include "bvh_measure.h"

#include "../ik/hardcodedProblems_inverseKinematics.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

            previousSolution->motion[0]=0;
            previousSolution->motion[1]=0;
            previousSolution->motion[2]=distance;
            
            penultimateSolution->motion[0]=0;
            penultimateSolution->motion[1]=0;
            penultimateSolution->motion[2]=distance;

            solution->motion[0]=0;
            solution->motion[1]=0;
            solution->motion[2]=distance;


            groundTruth->motion[0]=0;
            groundTruth->motion[1]=0;
            groundTruth->motion[2]=distance;


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
                    ikConfig.dumpScreenshots = 1;
                    ikConfig.maximumAcceptableStartingLoss=0.0; // Dont use this
                    ikConfig.verbose = 0;
                    ikConfig.tryMaintainingLocalOptima=1; //Less Jittery but can be stuck at local optima
                    ikConfig.ikVersion = IK_VERSION;
                    //------------------------------------


    
                      struct ikProblem * problem= (struct ikProblem * ) malloc(sizeof(struct ikProblem));
                      if (problem!=0)
                                     { memset(problem,0,sizeof(struct ikProblem)); } else
                                     { fprintf(stderr,"Failed to allocate memory for our IK problem..\n");  return 0; }

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
                               fprintf(stderr,"Could not prepare the problem for IK solution\n");
                               free(problem);
                               return 0;
                         }
                   
                   
                   const int MAX_ITERATIONS=15;
                   float fpsResult[MAX_ITERATIONS];
                   float accResult[MAX_ITERATIONS];
                   
                   
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

                        fprintf(stderr,"MAE in 2D Pixels went from %0.2f to %0.2f \n",initialMAEInPixels,finalMAEInPixels);
                        fprintf(stderr,"MAE in 3D mm went from %0.2f to %0.2f \n",initialMAEInMM*10,finalMAEInMM*10);
                        fprintf(stderr,"Computation time was %lu microseconds ( %0.2f fps )\n",endTime-startTime,convertStartEndTimeFromMicrosecondsToFPSIK(startTime,endTime));
                        
                        
                        fpsResult[ikConfig.iterations]=convertStartEndTimeFromMicrosecondsToFPSIK(startTime,endTime);
                        accResult[ikConfig.iterations]=finalMAEInMM*10;
                        fprintf(stderr,"%u/%u iteration / %0.2f MM / %0.2f \n",ikConfig.iterations,MAX_ITERATIONS,accResult[ikConfig.iterations],fpsResult[ikConfig.iterations]);
                        
                    }
                    else
                    {
                        fprintf(stderr,"Failed to run IK code..\n");
                    }
                   }
                   
                   
                   fprintf(stderr,"Results for gnuplot :)\n"); 
                   //fprintf(stdout,"Iterations,MAE,FPS\n"); 
                   for (int i=MAX_ITERATIONS; i>0; i--)
                   {
                        fprintf(stdout,"%u %0.2f %0.2f\n",i,accResult[i],fpsResult[i]); 
                   }
                   
                   
                  //Cleanup allocations needed for the problem..
                  cleanProblem(problem);
                  free(problem); 
                }
                else
                {
                    fprintf(stderr,"Could not project 2D points of target..\n");
                }
            }
        }
        freeMotionBuffer(&previousSolution);
        freeMotionBuffer(&solution);
        freeMotionBuffer(&initialSolution);
        freeMotionBuffer(&groundTruth);
    }

    return result;
}



