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
        fprintf(stderr,"Loaded %s => width:%u / height:%u / channels:%u / bitsperpixel:%u \n",filename,image->width,image->height,image->channels,image->bitsperpixel);
        if ( tri_paintModelUsingTexture(model,image->pixels,image->width,image->height,image->bitsperpixel,image->channels) )
        {
            fprintf(stderr,"Successfully painted TRI model using %s texture\n",filename);
            success=1;
        }
        destroyImage(image);
    }
    return success;
}

int textureLoadAndPack(struct TRI_Model * model,char * filename)
{
    int success=0;
    struct Image * image = readImage(filename,PNG_CODEC,0);
    if (image!=0)
    {
        fprintf(stderr,"Loaded %s => width:%u / height:%u / channels:%u / bitsperpixel:%u \n",filename,image->width,image->height,image->channels,image->bitsperpixel);
        if ( tri_packTextureInModel(model,image->pixels,image->width,image->height,image->bitsperpixel,image->channels) )
        {
            fprintf(stderr,"Successfully packed RGB image in TRI model using %s texture\n",filename);
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
        if (strcmp(argv[i],"--merge")==0)
        {
            inputFile  = argv[i+1];
            outputFile = argv[i+2];
            fprintf(stderr,GREEN "Merge multi Input Conversion from input(%s) to output(%s)\n" NORMAL,inputFile,outputFile);
            if ( (strstr(inputFile,".dae")!=0) || (strstr(inputFile,".obj")!=0) )
            {
                struct TRI_Container triContainer={0};
                if ( convertAssimpToTRIContainer(inputFile,&triContainer) )
                {
                    for (unsigned int meshID=0; meshID<triContainer.header.numberOfMeshes; meshID++)
                    {
                        fprintf(stderr,"Flattening mesh %u / %u \n",meshID+1,triContainer.header.numberOfMeshes);
                        fillFlatModelTriFromIndexedModelTri(&triContainer.mesh[meshID],&triContainer.mesh[meshID]);
                    }

                    if (!tri_simpleMergeOfTRIInContainer(originalModel,&triContainer))
                    {
                        fprintf(stderr,RED "Error merging multiple input in a single file..\n" NORMAL);
                        return 1;
                    }
                }
            }
        }
        else if (strcmp(argv[i],"--convert")==0)
        {
            inputFile  = argv[i+1];
            outputFile = argv[i+2];
            fprintf(stderr,GREEN "Converting input(%s) to output(%s)\n" NORMAL,inputFile,outputFile);
            if ( (strstr(inputFile,".fbx")!=0) || (strstr(inputFile,".dae")!=0) || (strstr(inputFile,".obj")!=0) )
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
        else if (strcmp(argv[i],"--droptexturealpha")==0)
        {
            tri_dropAlphaFromTexture(originalModel);
        }
        else if (strcmp(argv[i],"--applytexture")==0)
        {
            textureLoadAndPaint(originalModel,argv[i+1]);
        }
        else if (strcmp(argv[i],"--packtexture")==0)
        {
            textureLoadAndPack(originalModel,argv[i+1]);
        }
        else if (strcmp(argv[i],"--removeprefix")==0)
        {
            fprintf(stderr,GREEN "Remove Prefix %s\n" NORMAL,argv[i+1]);
            tri_removePrefixFromAllBoneNames(originalModel,argv[i+1]);
        }
        else if (strcmp(argv[i],"--paint")==0)
        {
            int r = atoi(argv[i+1]);
            int g = atoi(argv[i+2]);
            int b = atoi(argv[i+3]);

            fprintf(stderr,"Will paint mesh (RGB) (%u/%u/%u) \n",r,g,b);
            tri_paintModel(
                originalModel,
                r,
                g,
                b
            );
        }
    }

    if (outputFile!=0)
    {
        tri_saveModel(outputFile, originalModel);
    }


    for (int i=0; i<argc; i++)
    {
        if (strcmp(argv[1],"--test")==0)
        {
            struct TRI_Model *reloadedModel=  allocateModelTri();
            tri_loadModel( outputFile, reloadedModel);
            tri_saveModel( "resave.tri", reloadedModel);

            tri_freeModel(reloadedModel);
        }
    }

    tri_freeModel(flatModel);
    tri_freeModel(originalModel);



return 0;

}
