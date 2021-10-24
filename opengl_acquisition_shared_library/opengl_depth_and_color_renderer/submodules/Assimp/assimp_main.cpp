#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../src/Library/ModelLoader/model_loader_tri.h"

#include "assimp_loader.h"
#include "assimp_bvh.h"
#include "../../../../tools/Codecs/codecs.h"


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */



int textureLoadAndPaint(struct TRI_Model * model,char * filename)
{
    int success=0;
    struct Image * image = readImage(filename,PNG_CODEC,0);
    if (image!=0)
    {
        fprintf(stderr,"Loaded %s => width:%u / height:%u \n",filename,image->width,image->height);
        if ( paintTRIUsingTexture(model,image->pixels,image->width,image->height,image->bitsperpixel,image->channels) )
        {
            fprintf(stderr,"Successfully painted TRI model using %s texture\n",filename);
            success=1;
        }
        destroyImage(image);
    }
    return success;
}



int main (int argc, char *argv[])
{
    if (argc<4)
    {
        fprintf(stderr,"Not enough arguments , use as : \nassimpTester --convert source.dae target.tri \n");
        return 0;
    }


    fprintf(stderr,"assimpTester %s %s %s \n",argv[1],argv[2],argv[3]);

    char * inputFile = 0;
    char * outputFile = 0;

    struct TRI_Model * flatModel    =  allocateModelTri();
    struct TRI_Model * originalModel=  allocateModelTri();

    int selectMesh=0;



    for (int i=0; i<argc; i++)
    {
        if (strcmp(argv[i],"--convert")==0)
        {
            inputFile  = argv[i+1];
            outputFile = argv[i+2];
            fprintf(stderr,GREEN "Converting input(%s) to output(%s)\n" NORMAL,inputFile,outputFile);
            if ( (strstr(inputFile,".dae")!=0) || (strstr(inputFile,".obj")!=0) )
            {
                convertAssimpToTRI(inputFile,flatModel,originalModel,selectMesh);
            }
             else
            if (strstr(argv[2],".bvh")!=0)
            {
               doBVHConversion(argv[2]);
               fprintf(stderr,GREEN "Halting after BVH conversion\n" NORMAL);
               return 0;
            }
        }
        else if (strcmp(argv[i],"--mesh")==0)
        {
            selectMesh = atoi(argv[i+1]);
            fprintf(stderr,GREEN "Selecting mesh %u \n" NORMAL,selectMesh);
        }
        else if (strcmp(argv[i],"--applytexture")==0)
        {
            textureLoadAndPaint(originalModel,argv[i+1]);
        }
        else if (strcmp(argv[i],"--paint")==0)
        {
            int r = atoi(argv[i+1]);
            int g = atoi(argv[i+2]);
            int b = atoi(argv[i+3]);

            fprintf(stderr,"Will paint mesh (RGB) (%u/%u/%u) \n",r,g,b);
            paintTRI(
                originalModel,
                r,
                g,
                b
            );
        }
    }

    if (outputFile!=0)
    {
        saveModelTri(outputFile, originalModel);
    }


    for (int i=0; i<argc; i++)
    {
        if (strcmp(argv[1],"--test")==0)
        {
            struct TRI_Model *reloadedModel=  allocateModelTri();
            loadModelTri( outputFile, reloadedModel);
            saveModelTri( "resave.tri", reloadedModel);

            freeModelTri(reloadedModel);
        }
    }

    freeModelTri(flatModel);
    freeModelTri(originalModel);



return 0;

}
