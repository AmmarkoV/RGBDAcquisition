#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Include GLEW
#include <GL/glew.h>

//GLU
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include <time.h>
#include <string.h>

#include "../TrajectoryParser/InputParser_C.h"

#include "../Tools/save_to_file.h"
#include "../Tools/tools.h"

#include "../ModelLoader/hardcoded_shapes.h"
#include "../ModelLoader/model_loader_tri.h"
#include "../ModelLoader/model_loader_transform_joints.h"

#include "../ModelLoader/tri_bvh_controller.h"

#include "../MotionCaptureLoader/bvh_loader.h"
#include "../MotionCaptureLoader/edit/bvh_rename.h"
#include "../MotionCaptureLoader/edit/bvh_randomize.h"

#include "glx3.h"

#include "../../../../../tools/AmMatrix/matrix4x4Tools.h"
#include "../../../../../tools/AmMatrix/matrixOpenGL.h"

#include "../../../../../tools/Codecs/ppmInput.h"

#include "../Rendering/ShaderPipeline/shader_loader.h"
#include "../Rendering/ShaderPipeline/render_buffer.h"
#include "../Rendering/ShaderPipeline/uploadGeometry.h"
#include "../Rendering/ShaderPipeline/uploadTextures.h"
#include "../Rendering/downloadFromRenderer.h"

//Colored console output..
#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */

//Change this to change MultiRendering numbers..
#define originalWIDTH 1080
#define originalHEIGHT 1080
#define tilesToDoX 1
#define tilesToDoY 1
#define shrinkingFactor 1
//--------------------------------------------

float lastFramerate = 60;
unsigned long lastRenderingTime = 0;
unsigned int framesRendered = 0;

float backgroundColor[3]={0};
char flashTexturePixels = 0;
char dump2DPointOutput = 0;
unsigned int numberOfUniqueColors = 0;
unsigned char flashR = 255;
unsigned char flashG = 255;
unsigned char flashB = 0;

char renderEyes = 1;
char renderEyeHair = 1;
char renderHair = 0;

char VSYNC = 0;
int performBoneTransformsInCPU = 0; // <- Experimental when 0

//Virtual Camera Intrinsics
float fX = 1235.423889;
float fY = 1233.48468;
float nearPlane = 0.1;
float farPlane  = 1000.0;

struct textureAssociation
{
   unsigned int x;
   unsigned int y;
};

struct pose6D
 {
     float x,y,z;
     float roll,pitch,yaw;

     char usePoseMatrixDirectly;
     struct Matrix4x4OfFloats m;
 };


struct GPUTriModel
{
  //------------------------
  GLuint programID;
  //------------------------
  struct shaderModelData shader;
  //------------------------
  struct TRI_Model * model;
};

int windowSizeUpdated(unsigned int newWidth , unsigned int newHeight)
{
    return 0;
}

int handleUserInput(char key,int state,unsigned int x, unsigned int y)
{
    return 0;
}


unsigned int decodeUniqueColor(unsigned int totalColors,unsigned char r,unsigned char g,unsigned char b)
{
  unsigned int colorStep = (255*255*255) / (1+totalColors);
  //------------------------------------
  unsigned int thisColor = b + (256 * g) + (256 * 256 * r);
  unsigned int pointID = thisColor / colorStep;
  //------------------------------------
  return pointID;
}


void encodeUniqueColor(unsigned int colorNumber,unsigned int totalColors,unsigned char * r,unsigned char * g,unsigned char * b)
{
  unsigned int colorStep = (255*255*255) / (1+totalColors);
  unsigned int thisColor = colorNumber * colorStep;
  //------------------------------------
  *r = (thisColor & 0xFF0000)>>16;
  *g = (thisColor & 0x00FF00)>>8;
  *b = (thisColor & 0x0000FF);
  //------------------------------------
}


unsigned int  * readKeyPoint(const char * filename,char flipX,unsigned int width,unsigned int height,unsigned int * outputNumberOfPoints)
{
  unsigned int * m = 0;
  unsigned int numberOfPoints = 0;
  FILE * fp = fopen(filename,"r");

  if (fp!=0)
  {
      fscanf(fp,"%u\n",&numberOfPoints);

      m = (unsigned int  *) malloc(sizeof(unsigned int ) * numberOfPoints * 2);

      for (int i=0; i<numberOfPoints; i++)
      {
        float x,y;
        fscanf(fp,"%f\n",&x);
        fscanf(fp,"%f\n",&y);

        if (flipX)
        {
          x = 1 - x;
        }

        m[i*2 + 0] = (unsigned int) (x * width); //< - NOTICE X FLIP!
        m[i*2 + 1] = (unsigned int) (y * height);
      }

      fclose(fp);
  }

  *outputNumberOfPoints = numberOfPoints;
  return m;
}

void paintRGBPixel(unsigned char * image,unsigned int imageWidth,unsigned int imageHeight,unsigned int x,unsigned int y,unsigned char r,unsigned char g,unsigned char b)
{
  if (image!=0)
  {
    if ( (x<imageWidth) && (y<imageHeight) )
    {
      unsigned char * target = image + (imageWidth * y * 3) + (x * 3);
      *target=r;   target++;
      *target=g;   target++;
      *target=b;   target++;
    }
  }
}



void parseTextureToScreenAssociations(const char * filename,const char * faceFilename,struct TRI_Model * indexedHumanModel,struct TRI_Model * indexedEyeModel)
{
  #define FADE_TO_BLACK 1
  FILE * fp = fopen(filename,"r");

  struct InputParserC * ipc = InputParser_Create(8096,6);
  InputParser_SetDelimeter(ipc,0,',');
  InputParser_SetDelimeter(ipc,1,'(');
  InputParser_SetDelimeter(ipc,2,')');
  InputParser_SetDelimeter(ipc,3,0);
  InputParser_SetDelimeter(ipc,4,10);
  InputParser_SetDelimeter(ipc,5,13);

  unsigned char r,g,b;

  unsigned int numberOfPoints = 0;
  unsigned int  * keypoints = readKeyPoint(faceFilename,1,originalWIDTH,originalHEIGHT,&numberOfPoints);
  numberOfUniqueColors = numberOfPoints + 2; // We also allocated 2 more points for l/r eyes!

  if (keypoints == 0) { return ; }

  if (fp!=0)
  {
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    unsigned char * rgb =  (unsigned char * ) malloc(sizeof(unsigned char) * originalWIDTH * originalHEIGHT *3);

    #if FADE_TO_BLACK
      memset(indexedEyeModel->textureData,0,sizeof(char) * indexedEyeModel->header.textureDataWidth * indexedEyeModel->header.textureDataHeight * indexedEyeModel->header.textureDataChannels);
      //Hardcoded eye positions 300,726 and 726,300
      encodeUniqueColor(numberOfPoints,numberOfUniqueColors,&r,&g,&b);
      paintRGBPixel(
                    indexedEyeModel->textureData,
                    indexedEyeModel->header.textureDataWidth,
                    indexedEyeModel->header.textureDataHeight,
                    300,726,
                    r,g,b
                   );
      encodeUniqueColor(numberOfPoints+1,numberOfUniqueColors,&r,&g,&b);
      paintRGBPixel(
                    indexedEyeModel->textureData,
                    indexedEyeModel->header.textureDataWidth,
                    indexedEyeModel->header.textureDataHeight,
                    726,300,
                    r,g,b
                   );
      //----------------------------------------------


      memset(indexedHumanModel->textureData,0,sizeof(char) * indexedHumanModel->header.textureDataWidth * indexedHumanModel->header.textureDataHeight * indexedHumanModel->header.textureDataChannels);
      memset(rgb,0,sizeof(char) * originalWIDTH * originalHEIGHT * 3);
    #endif // FADE_TO_BLACK


    struct textureAssociation * mappingFbToTex = (struct textureAssociation *) malloc(sizeof(struct textureAssociation) * originalWIDTH * originalHEIGHT);

    while ((read = getline(&line, &len, fp)) != -1)
        {
          unsigned int numberOfFields = InputParser_SeperateWords(ipc,line,1);
          //------------------------------------------------------------------
          unsigned int x        = InputParser_GetWordInt(ipc,1);
          unsigned int y        = InputParser_GetWordInt(ipc,2);
          unsigned int textureX = InputParser_GetWordInt(ipc,3);
          unsigned int textureY = InputParser_GetWordInt(ipc,4);

          #if FADE_TO_BLACK==0
           //Show all active points to debug..
           unsigned char * tx = indexedHumanModel->textureData + (indexedHumanModel->header.textureDataWidth * textureY * 3) + (textureX * 3);
           *tx=0;   tx++;
           *tx=64; tx++;
           *tx=64; tx++;

           unsigned char * rgbPtr = rgb + (y * originalWIDTH * 3) + (x * 3);
           *rgbPtr = 0; rgbPtr++;
           *rgbPtr = 64; rgbPtr++;
           *rgbPtr = 64; rgbPtr++;
          #endif // FADE_TO_BLACK

          //------------------------------------------------------------------
          if ( (x!=0) || (y!=0) || (textureX!=0) || (textureY!=0) )
            {
              //fprintf(stderr,"X=%u,Y=%u -> tX=%u,tY=%u \n",x,y,textureX,textureY);
              unsigned int ptr = (y * originalWIDTH) + x;
              mappingFbToTex[ptr].x = textureX;
              mappingFbToTex[ptr].y = textureY;
            }
        }

   if (line!=0) { free(line); }
   fclose(fp);



   for (int i=0; i<numberOfPoints; i++)
   {
      unsigned int textureX = keypoints[i*2+0];
      unsigned int textureY = keypoints[i*2+1];
      //---------------------------------------

      unsigned int ptr = (textureY * originalWIDTH) + textureX;
      //mappingFbToTex[ptr].x = textureX;
      //mappingFbToTex[ptr].y = textureY;
      fprintf(stderr," tX=%u,tY=%u -> X=%u,Y=%u\n",textureX,textureY,mappingFbToTex[ptr].x,mappingFbToTex[ptr].y);

      unsigned char r,g,b;
      encodeUniqueColor(i,numberOfUniqueColors,&r,&g,&b);

      unsigned int doubleCheckedPoint = decodeUniqueColor(numberOfUniqueColors,r,g,b);

      if (i!=doubleCheckedPoint)
      {
          fprintf(stderr,RED "Mismatch (%u,%u,%u) point %u  decoded as %u\n" NORMAL,r,g,b,i,doubleCheckedPoint);
      }

      unsigned char * tx = indexedHumanModel->textureData + (indexedHumanModel->header.textureDataWidth * mappingFbToTex[ptr].y * 3) + (mappingFbToTex[ptr].x * 3);
      *tx=r;   tx++;
      *tx=g;   tx++;
      *tx=b;   tx++;

      if (rgb!=0)
      {
       unsigned char * rgbPtr = rgb + (textureY * originalWIDTH * 3) + (textureX * 3);
       *rgbPtr = r; rgbPtr++;
       *rgbPtr = g; rgbPtr++;
       *rgbPtr = b; rgbPtr++;
      }
   }

   saveRawImageToFileOGLR("occupiedTexture.pnm",indexedHumanModel->textureData,indexedHumanModel->header.textureDataWidth,indexedHumanModel->header.textureDataHeight,3,8);
   saveRawImageToFileOGLR("occupiedEyesTexture.pnm",indexedEyeModel->textureData,indexedEyeModel->header.textureDataWidth,indexedEyeModel->header.textureDataHeight,3,8);

   if (rgb!=0)
      {
       saveRawImageToFileOGLR("occupation.pnm",rgb,originalWIDTH,originalHEIGHT,3,8);
       free(rgb);
      }

   free(keypoints);
   InputParser_Destroy(ipc);
  }

}

int drawObjectAT(
                 GLuint programID,
                 GLuint vao,
                 GLuint MatrixID,
                 GLuint TextureID,
                 unsigned int triangleCount,
                 unsigned int elementCount,
                 //-----------------------------------------
                 float x,
                 float y,
                 float z,
                 float roll,
                 float pitch,
                 float yaw,
                 //-----------------------------------------
                 struct Matrix4x4OfFloats * projectionMatrix,
                 struct Matrix4x4OfFloats * viewportMatrix,
                 struct Matrix4x4OfFloats * viewMatrix,
                 //-----------------------------------------
                 char renderWireframe
                 )
{
       struct Matrix4x4OfFloats modelMatrix;
       create4x4FModelTransformation(
                                     &modelMatrix,
                                     //Rotation Component
                                     roll,//roll
                                     pitch ,//pitch
                                     yaw ,//yaw
                                     ROTATION_ORDER_RPY,
                                     //-----------------------------------------
                                     //Translation Component (XYZ)
                                     (float) x,
                                     (float) y,
                                     (float) z,
                                     //-----------------------------------------
                                     -1.0,//scaleX,
                                     1.0,//scaleY,
                                     -1.0//scaleZ
                                    );

       return drawVertexArrayWithMVPMatrices(
                                             programID,
                                             vao,
                                             MatrixID,
                                             TextureID,
                                             triangleCount,
                                             elementCount,
                                             //-----------------------------------------
                                             &modelMatrix,
                                             //-----------------------------------------
                                             projectionMatrix,
                                             viewportMatrix,
                                             viewMatrix,
                                             //-----------------------------------------
                                             renderWireframe
                                            );
}




int doOGLSingleDrawing(
                        int programID,
                        GLuint MVPMatrixID,
                        GLuint TextureID,
                        struct pose6D * pose,
                        GLuint VAO,
                        unsigned int triangleCount,
                        unsigned int elementCount,
                        unsigned int width,
                        unsigned int height
                      )
{
  struct Matrix4x4OfFloats projectionMatrix;
  struct Matrix4x4OfFloats viewportMatrix;
  struct Matrix4x4OfFloats viewMatrix;

  prepareRenderingMatrices(
                              fX, //fx
                              fY, //fy
                              0.0,        //skew
                              (float) width/2,    //cx
                              (float) height/2,   //cy
                              (float) width,      //Window Width
                              (float) height,     //Window Height
                              nearPlane,     //Near
                              farPlane,      //Far
                              &projectionMatrix,
                              &viewMatrix,
                              &viewportMatrix
                         );

  unsigned int viewportWidth = (unsigned int) width;
  unsigned int viewportHeight = (unsigned int) height;
  glViewport(0,0,viewportWidth,viewportHeight);

  float newViewport[4]={0,0,viewportWidth,viewportHeight};
  float projectionMatrixViewportCorrected[16];
  correctProjectionMatrixForDifferentViewport(
                                               projectionMatrixViewportCorrected,
                                               projectionMatrix.m,
                                               viewportMatrix.m,
                                               newViewport
                                             );
  //-------------------------------
  //-------------------------------
  //-------------------------------
  if (pose->usePoseMatrixDirectly)
     {
      drawVertexArrayWithMVPMatrices(
                                     programID,
                                     VAO,
                                     MVPMatrixID,
                                     TextureID,
                                     triangleCount,
                                     elementCount,
                                     //-------------
                                     &pose->m,
                                     //-------------
                                     &projectionMatrix,
                                     &viewportMatrix,
                                     &viewMatrix,
                                     0 //Wireframe
                                    );
     } else
     {
      drawObjectAT(
                  programID,
                  VAO,
                  MVPMatrixID,
                  TextureID,
                  triangleCount,
                  elementCount,
                  //-------------
                  pose->x,
                  pose->y,
                  pose->z,
                  pose->roll,
                  pose->pitch,
                  pose->yaw,
                  //-------------
                  &projectionMatrix,
                  &viewportMatrix,
                  &viewMatrix,
                  0 //Wireframe
                  );
     }
  //-------------------------------
  //-------------------------------
  //-------------------------------

  return 1;
}






int doOGLDrawing(
                 int programID,
                 GLuint MVPMatrixID,
                 //------------------
                 struct GPUTriModel * gpuEyelashes,
                 struct GPUTriModel * gpuEyebrows,
                 struct GPUTriModel * gpuHair,
                 struct GPUTriModel * gpuEyes,
                 struct GPUTriModel * gpuHuman,
                 //------------------
                 struct pose6D * humanPose,
                 //------------------
                 unsigned int width,
                 unsigned int height
                )
{
  struct Matrix4x4OfFloats projectionMatrix;
  struct Matrix4x4OfFloats viewportMatrix;
  struct Matrix4x4OfFloats viewMatrix;

  prepareRenderingMatrices(
                              fX,  //fx
                              fY,  //fy
                              0.0,        //skew
                              (float) width/2,    //cx
                              (float) height/2,   //cy
                              (float) width,      //Window Width
                              (float) height,     //Window Height
                              nearPlane,     //Near
                              farPlane,      //Far
                              &projectionMatrix,
                              &viewMatrix,
                              &viewportMatrix
                         );

  unsigned int viewportWidth = (unsigned int) width;
  unsigned int viewportHeight = (unsigned int) height;
  glViewport(0,0,viewportWidth,viewportHeight);

  float newViewport[4]={0,0,viewportWidth,viewportHeight};
  float projectionMatrixViewportCorrected[16];
  correctProjectionMatrixForDifferentViewport(
                                               projectionMatrixViewportCorrected,
                                               projectionMatrix.m,
                                               viewportMatrix.m,
                                               newViewport
                                             );
  //-------------------------------
  //-------------------------------
  //-------------------------------
  if (humanPose->usePoseMatrixDirectly)
     {
      if (renderHair)
      {
      drawVertexArrayWithMVPMatrices(
                                     programID,
                                     gpuHair->shader.VAO,
                                     MVPMatrixID,
                                     gpuHair->model->header.textureBindGLBuffer,
                                     gpuHair->shader.triangleCount,
                                     gpuHair->model->header.numberOfIndices,
                                     //-------------
                                     &humanPose->m,
                                     //-------------
                                     &projectionMatrix,
                                     &viewportMatrix,
                                     &viewMatrix,
                                     0 //Wireframe
                                    );
      }

      if (renderEyes)
      {
          if (renderEyeHair)
          {
      drawVertexArrayWithMVPMatrices(
                                     programID,
                                     gpuEyelashes->shader.VAO,
                                     MVPMatrixID,
                                     gpuEyelashes->model->header.textureBindGLBuffer,
                                     gpuEyelashes->shader.triangleCount,
                                     gpuEyelashes->model->header.numberOfIndices,
                                     //-------------
                                     &humanPose->m,
                                     //-------------
                                     &projectionMatrix,
                                     &viewportMatrix,
                                     &viewMatrix,
                                     0 //Wireframe
                                    );
      drawVertexArrayWithMVPMatrices(
                                     programID,
                                     gpuEyebrows->shader.VAO,
                                     MVPMatrixID,
                                     gpuEyebrows->model->header.textureBindGLBuffer,
                                     gpuEyelashes->shader.triangleCount,
                                     gpuEyelashes->model->header.numberOfIndices,
                                     //-------------
                                     &humanPose->m,
                                     //-------------
                                     &projectionMatrix,
                                     &viewportMatrix,
                                     &viewMatrix,
                                     0 //Wireframe
                                    );
          }

      drawVertexArrayWithMVPMatrices(
                                     programID,
                                     gpuEyes->shader.VAO,
                                     MVPMatrixID,
                                     gpuEyes->model->header.textureBindGLBuffer,
                                     gpuEyes->shader.triangleCount,
                                     gpuEyes->model->header.numberOfIndices,
                                     //-------------
                                     &humanPose->m,
                                     //-------------
                                     &projectionMatrix,
                                     &viewportMatrix,
                                     &viewMatrix,
                                     0 //Wireframe
                                    );

      }

      drawVertexArrayWithMVPMatrices(
                                     programID,
                                     gpuHuman->shader.VAO,
                                     MVPMatrixID,
                                     gpuHuman->model->header.textureBindGLBuffer,
                                     gpuHuman->shader.triangleCount,
                                     gpuHuman->model->header.numberOfIndices,
                                     //-------------
                                     &humanPose->m,
                                     //-------------
                                     &projectionMatrix,
                                     &viewportMatrix,
                                     &viewMatrix,
                                     0 //Wireframe
                                    );
     } else
     {
      if (renderHair)
      {
      drawObjectAT(
                  programID,
                  gpuHair->shader.VAO,
                  MVPMatrixID,
                  gpuHair->model->header.textureBindGLBuffer,
                  gpuHair->shader.triangleCount,
                  gpuHair->model->header.numberOfIndices,
                  //-------------
                  humanPose->x,
                  humanPose->y,
                  humanPose->z,
                  humanPose->roll,
                  humanPose->pitch,
                  humanPose->yaw,
                  //-------------
                  &projectionMatrix,
                  &viewportMatrix,
                  &viewMatrix,
                  0 //Wireframe
                  );
      }

      if (renderEyes)
      {
       if (renderEyeHair)
          {
      drawObjectAT(
                  programID,
                  gpuEyelashes->shader.VAO,
                  MVPMatrixID,
                  gpuEyelashes->model->header.textureBindGLBuffer,
                  gpuEyelashes->shader.triangleCount,
                  gpuEyelashes->model->header.numberOfIndices,
                  //-------------
                  humanPose->x,
                  humanPose->y,
                  humanPose->z,
                  humanPose->roll,
                  humanPose->pitch,
                  humanPose->yaw,
                  //-------------
                  &projectionMatrix,
                  &viewportMatrix,
                  &viewMatrix,
                  0 //Wireframe
                  );
      drawObjectAT(
                  programID,
                  gpuEyebrows->shader.VAO,
                  MVPMatrixID,
                  gpuEyebrows->model->header.textureBindGLBuffer,
                  gpuEyebrows->shader.triangleCount,
                  gpuEyebrows->model->header.numberOfIndices,
                  //-------------
                  humanPose->x,
                  humanPose->y,
                  humanPose->z,
                  humanPose->roll,
                  humanPose->pitch,
                  humanPose->yaw,
                  //-------------
                  &projectionMatrix,
                  &viewportMatrix,
                  &viewMatrix,
                  0 //Wireframe
                  );
          }
      drawObjectAT(
                  programID,
                  gpuEyes->shader.VAO,
                  MVPMatrixID,
                  gpuEyes->model->header.textureBindGLBuffer,
                  gpuEyes->shader.triangleCount,
                  gpuEyes->model->header.numberOfIndices,
                  //-------------
                  humanPose->x,
                  humanPose->y,
                  humanPose->z,
                  humanPose->roll,
                  humanPose->pitch,
                  humanPose->yaw,
                  //-------------
                  &projectionMatrix,
                  &viewportMatrix,
                  &viewMatrix,
                  0 //Wireframe
                  );
      }

      drawObjectAT(
                  programID,
                  gpuHuman->shader.VAO,
                  MVPMatrixID,
                  gpuHuman->model->header.textureBindGLBuffer,
                  gpuHuman->shader.triangleCount,
                  gpuHuman->model->header.numberOfIndices,
                  //-------------
                  humanPose->x,
                  humanPose->y,
                  humanPose->z,
                  humanPose->roll,
                  humanPose->pitch,
                  humanPose->yaw,
                  //-------------
                  &projectionMatrix,
                  &viewportMatrix,
                  &viewMatrix,
                  0 //Wireframe
                  );
     }
  //-------------------------------
  //-------------------------------
  //-------------------------------

  return 1;
}

int initializeOGLRenderer(
                           GLuint * programID,
                           GLuint * programFrameBufferID,
                           GLuint * FramebufferName,
                           GLuint * renderedTexture,
                           GLuint * renderedDepth,
                           int skinnedRendering,
                           unsigned int WIDTH,
                           unsigned int HEIGHT
                         )
{
	// Create and compile our GLSL program from the shaders
	//struct shaderObject * sho = loadShader("../../../shaders/TransformVertexShader.vertexshader", "../../../shaders/ColorFragmentShader.fragmentshader");
	struct shaderObject * sho = 0;

	if (skinnedRendering) { sho = loadShader("../../../shaders/skeletonWithTexture.vert", "../../../shaders/skeletonWithTexture.frag"); } else
	                      { sho = loadShader("../../../shaders/simpleWithTexture.vert"  , "../../../shaders/simpleWithTexture.frag");   }

	if (sho==0) {  checkOpenGLError(__FILE__, __LINE__); return 0; }

	struct shaderObject * textureFramebuffer = loadShader("../../../shaders/virtualFramebuffer.vert", "../../../shaders/virtualFramebuffer.frag");
    if (textureFramebuffer==0) {  checkOpenGLError(__FILE__, __LINE__); return 0; }

    *programID = sho->ProgramObject;
    *programFrameBufferID = textureFramebuffer->ProgramObject;

	// Use our shader
	glUseProgram(*programID);

	// Black background
	glClearColor(backgroundColor[0],backgroundColor[1],backgroundColor[2],0.0f);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);

	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);

    initializeFramebuffer(FramebufferName,renderedTexture,renderedDepth,WIDTH,HEIGHT);

    fprintf(stderr,"Ready to start pushing geometry  ");
    return 1;
}

void processGPUTRI(struct GPUTriModel * gputri)
{
    struct TRI_Model * model = gputri->model;
    if (model!=0)
    {
     gputri->shader.triangleCount       = (unsigned int) model->header.numberOfVertices/3;
     //-----------------------------------------------------------------------------
     gputri->shader.vertices            = model->vertices;
     gputri->shader.sizeOfVertices      = model->header.numberOfVertices * sizeof(float);
     //-----------------------------------------------------------------------------
     gputri->shader.normals             = model->normal;
     gputri->shader.sizeOfNormals       = model->header.numberOfNormals * sizeof(float);
     //-----------------------------------------------------------------------------
     gputri->shader.textureCoords       = model->textureCoords;
     gputri->shader.sizeOfTextureCoords = model->header.numberOfTextureCoords * sizeof(float);
     //-----------------------------------------------------------------------------
     gputri->shader.colors              = model->colors;
     gputri->shader.sizeOfColors        = model->header.numberOfColors * sizeof(float);
     //-----------------------------------------------------------------------------
     gputri->shader.indices             = model->indices;
     gputri->shader.sizeOfIndices       = model->header.numberOfIndices * sizeof(unsigned int);
     //-----------------------------------------------------------------------------

     //If anything below fails sizes of bones will be zero and
     //wont cause rendering problems
     gputri->shader.sizeOfBoneIndexes     = 0;
     gputri->shader.sizeOfBoneWeightValues= 0;
     gputri->shader.sizeOfBoneTransforms  = 0;
     //-----------------------------------------------------------------------------
     if ( (model->bones!=0) && (!performBoneTransformsInCPU) ) // If we have bones that need to be processed in the shader..
     {
       struct TRI_BonesPackagedPerVertex packed={0};

       //Since we cycle through the same data it is important to provide
       //the tri_packageBoneDataPerVertex call with our previous values (if any) so that
       //it manages the memory correctly
       packed.boneIndexes      =  gputri->shader.boneIndexes;
       packed.boneWeightValues =  gputri->shader.boneWeightValues;
       packed.boneTransforms   =  gputri->shader.boneTransforms;

       if ( tri_packageBoneDataPerVertex(&packed,model,gputri->shader.initialized) ) //<- this call repacks bones per vertex and is pretty heavy..
       {
        gputri->shader.numberOfBones          = packed.numberOfBones;
        gputri->shader.numberOfBonesPerVertex = packed.numberOfBonesPerVertex;
        //--------------------------------------------------------------------
        gputri->shader.boneIndexes            = packed.boneIndexes;
        gputri->shader.sizeOfBoneIndexes      = packed.sizeOfBoneIndexes;
        //--------------------------------------------------------------------
        gputri->shader.boneWeightValues       = packed.boneWeightValues;
        gputri->shader.sizeOfBoneWeightValues = packed.sizeOfBoneWeightValues;
        //--------------------------------------------------------------------
        gputri->shader.boneTransforms         = packed.boneTransforms;
        gputri->shader.sizeOfBoneTransforms   = packed.sizeOfBoneTransforms;
       }
     }
     //-----------------------------------------------------------------------------
    }
}


void deallocateGpuTRI(struct GPUTriModel * gputri)
{
 if (gputri->shader.initialized==1)
 {
	//-------------------------------------
    glDeleteBuffers(1, &gputri->shader.arrayBuffer);
	//-------------------------------------
	glDeleteBuffers(1, &gputri->shader.elementBuffer);
	//-------------------------------------
	glDeleteVertexArrays(1, &gputri->shader.VAO);
	//-------------------------------------
	gputri->shader.initialized=0;
	//-------------------------------------
    if (gputri->shader.boneIndexes!=0)
      {
         free(gputri->shader.boneIndexes);
         gputri->shader.boneIndexes=0;
      }
	//-------------------------------------
    if (gputri->shader.boneWeightValues!=0)
      {
         free(gputri->shader.boneWeightValues);
         gputri->shader.boneWeightValues=0;
      }
	//-------------------------------------
    if (gputri->shader.boneTransforms!=0)
     {
         free(gputri->shader.boneTransforms);
         gputri->shader.boneTransforms=0;
     }
	//-------------------------------------
 }
}

int doDrawing(
                GLuint programID,
                GLuint programFrameBufferID,
                GLuint FramebufferName,
                GLuint renderedTexture,
                GLuint renderedDepth,
                struct pose6D * humanPose,
                struct GPUTriModel * gpuEyelashes,
                struct GPUTriModel * gpuEyebrows,
                struct GPUTriModel * gpuHair,
                struct GPUTriModel * gpuEyes,
                struct GPUTriModel * gpuHuman,
                unsigned int WIDTH,
                unsigned int HEIGHT,
                int renderForever
             )
{
 	// Get a handle for our "MVP" uniform
	GLuint MVPMatrixID = glGetUniformLocation(programID, "MVP");
    //------------------------------------------------------------------------------------
    processGPUTRI(gpuEyelashes);
    gpuEyelashes->shader.initialized=
    pushBonesToBufferData(
                             (gpuEyelashes->shader.initialized==0),// generateNewVao
                             (gpuEyelashes->shader.initialized==0),// generateNewArrayBuffer
                             (gpuEyelashes->shader.initialized==0),// generateNewElementBuffer
                              gpuEyelashes->programID,
                             //-------------------------------------------------------------------
                              &gpuEyelashes->shader
                            );
    //------------------------------------------------------------------------------------
    processGPUTRI(gpuEyebrows);
    gpuEyebrows->shader.initialized=
    pushBonesToBufferData(
                             (gpuEyebrows->shader.initialized==0),// generateNewVao
                             (gpuEyebrows->shader.initialized==0),// generateNewArrayBuffer
                             (gpuEyebrows->shader.initialized==0),// generateNewElementBuffer
                              gpuEyebrows->programID,
                             //-------------------------------------------------------------------
                              &gpuEyebrows->shader
                            );
    //------------------------------------------------------------------------------------
    processGPUTRI(gpuHair);
    gpuHair->shader.initialized=
    pushBonesToBufferData(
                             (gpuHair->shader.initialized==0),// generateNewVao
                             (gpuHair->shader.initialized==0),// generateNewArrayBuffer
                             (gpuHair->shader.initialized==0),// generateNewElementBuffer
                              gpuHair->programID,
                             //-------------------------------------------------------------------
                              &gpuHair->shader
                            );
    //------------------------------------------------------------------------------------
    processGPUTRI(gpuEyes);
    gpuEyes->shader.initialized=
    pushBonesToBufferData(
                             (gpuEyes->shader.initialized==0),// generateNewVao
                             (gpuEyes->shader.initialized==0),// generateNewArrayBuffer
                             (gpuEyes->shader.initialized==0),// generateNewElementBuffer
                              gpuEyes->programID,
                             //-------------------------------------------------------------------
                              &gpuEyes->shader
                          );
    //------------------------------------------------------------------------------------
    processGPUTRI(gpuHuman);
    gpuHuman->shader.initialized=
    pushBonesToBufferData(
                             (gpuHuman->shader.initialized==0),// generateNewVao
                             (gpuHuman->shader.initialized==0),// generateNewArrayBuffer
                             (gpuHuman->shader.initialized==0),// generateNewElementBuffer
                              gpuHuman->programID,
                             //-------------------------------------------------------------------
                              &gpuHuman->shader
                          );
    //------------------------------------------------------------------------------------
	 GLuint quad_vertexbuffer=0;
	 glGenBuffers(1, &quad_vertexbuffer);
	 glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
	 glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);

	 //Uniforms
	 GLuint texID = glGetUniformLocation(programFrameBufferID, "renderedTexture");
	 GLuint timeID = glGetUniformLocation(programFrameBufferID, "iTime");
	 GLuint resolutionID = glGetUniformLocation(programFrameBufferID, "iResolution");

	 do
     {
        // Render to our framebuffer
		glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
		glViewport(0,0,WIDTH,HEIGHT); // Render on the whole framebuffer, complete from the lower left corner to the upper right

       //-----------------------------------------------
        if (framesRendered%10==0) { fprintf(stderr,"\r%0.2f FPS                                         \r", lastFramerate ); }
       //-----------------------------------------------

        //This works better
        //glClearColor(0.2,0.2,0.2,1);
	    glClearColor(backgroundColor[0],backgroundColor[1],backgroundColor[2],0.0f);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 		// Clear the screen

        doOGLDrawing(
                     programID,
                     MVPMatrixID,
                     //------------------------
                     gpuEyelashes,
                     gpuEyebrows,
                     gpuHair,
                     gpuEyes,
                     gpuHuman,
                     //------------------------
                     humanPose,
                     //------------------------
                     WIDTH,
                     HEIGHT
                    );

        //We have accumulated all data on the framebuffer and will now draw it back..
        drawFramebufferToScreen(
                                programFrameBufferID,
                                quad_vertexbuffer,
                                renderedTexture,
                                texID,
                                timeID,
                                resolutionID,
                                WIDTH,HEIGHT
                               );

		// Swap buffers
        glx3_endRedraw();


      //---------------------------------------------------------------
      //------------------- Calculate Framerate -----------------------
      //---------------------------------------------------------------
       unsigned long now=GetTickCountMicrosecondsIK();//GetTickCountMilliseconds();
       unsigned long elapsedTime=now-lastRenderingTime;
       if (elapsedTime==0) { elapsedTime=1; }
       lastFramerate = (float) 1000000/(elapsedTime);
       lastRenderingTime = now;
      //---------------------------------------------------------------
        ++framesRendered;
      //---------------------------------------------------------------

      if (renderForever) { usleep(1); } // Cap framerate if looping here...
	} // Check if the ESC key was pressed or the window was closed
    while(renderForever);

	// Cleanup VBO and shader
	glDeleteBuffers(1, &quad_vertexbuffer);
	//-------------------------------------
	//We do not need to create everything every time..
	//deallocateGpuTRI(gpuEyelashes);
	//deallocateGpuTRI(gpuEyebrows);
	//deallocateGpuTRI(gpuHair);
	//deallocateGpuTRI(gpuEyes);
	//deallocateGpuTRI(gpuHuman);
	return 1;
}





int doSkeletonDraw(
                   GLuint programID,
                   GLuint programFrameBufferID,
                   GLuint FramebufferName,
                   GLuint renderedTexture,
                   GLuint renderedDepth,
                   struct pose6D * humanPose,
                   struct TRI_Model * humanModel,
                   struct TRI_Model * axisModel,
                   unsigned int WIDTH,
                   unsigned int HEIGHT,
                   int renderForever
                  )
{
 	// Get a handle for our "MVP" uniform
	GLuint MVPMatrixID = glGetUniformLocation(programID, "MVP");
    //------------------------------------------------------------------------------------
    GLuint axisVAO=0;
    GLuint axisArrayBuffer=0;
    GLuint axisElementBuffer=0;
    unsigned int axisTriangleCount=0;

    int usePrimitive = 0;

    if (!usePrimitive)
    {
     axisTriangleCount  =  (unsigned int)  axisModel->header.numberOfVertices/3;
     pushObjectToBufferData(
                             1,
                             &axisVAO,
                             &axisArrayBuffer,
                             &axisElementBuffer,
                             programID,
                             axisModel->vertices       ,  axisModel->header.numberOfVertices      * sizeof(float),
                             axisModel->normal         ,  axisModel->header.numberOfNormals       * sizeof(float),
                             axisModel->textureCoords  ,  axisModel->header.numberOfTextureCoords * sizeof(float),      //0,0 //No Texture
                             axisModel->colors         ,  axisModel->header.numberOfColors        * sizeof(float),
                             axisModel->indices        ,  axisModel->header.numberOfIndices       * sizeof(unsigned int)//0,0 //Not Indexed
                           );
    } else
    {
     axisTriangleCount  = pyramidTriangleCount;
     pushObjectToBufferData(
                             1,
                             &axisVAO,
                             &axisArrayBuffer,
                             &axisElementBuffer,
                             programID,
                             pyramidCoords     ,  sizeof(pyramidCoords),
                             pyramidNormals    ,  sizeof(pyramidNormals),
                             pyramidTexCoords  ,  sizeof(pyramidTexCoords),      //0,0 //No Texture
                             cubeColors        ,  sizeof(cubeColors),
                             0                 ,  0//0,0 //Not Indexed
                          );
    }
    //------------------------------------------------------------------------------------
	 GLuint quad_vertexbuffer;
	 glGenBuffers(1, &quad_vertexbuffer);
	 glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
	 glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);

	 // Create and compile our GLSL program from the shaders
	 GLuint texID = glGetUniformLocation(programFrameBufferID, "renderedTexture");
	 // Create and compile our GLSL program from the shaders
	 GLuint timeID = glGetUniformLocation(programFrameBufferID, "iTime");
	 GLuint resolutionID = glGetUniformLocation(programFrameBufferID, "iResolution");

	 do
     {
        // Render to our framebuffer
		glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
		glViewport(0,0,WIDTH,HEIGHT); // Render on the whole framebuffer, complete from the lower left corner to the upper right

       //-----------------------------------------------
        if (framesRendered%10==0) { fprintf(stderr,"\r%0.2f FPS                                         \r", lastFramerate ); }
       //-----------------------------------------------

	    glClearColor(backgroundColor[0],backgroundColor[1],backgroundColor[2],1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 		// Clear the screen


        if (humanModel->bones==0)
        {
            fprintf(stderr,"doSkeletonDraw called with no bones!\n");
            return 0;
        }

          //Add an axis to help
          /*struct pose6D axisPose={0};
          axisPose.z=10;
          doOGLSingleDrawing(
                             programID,
                             MVPMatrixID,
                             0,
                             &axisPose,
                             axisVAO,
                             axisTriangleCount,
                             axisElementBuffer,
                             WIDTH,
                             HEIGHT
                           );*/


        fprintf(stderr,"BoneID %u -> %u \n",0,humanModel->header.numberOfBones);
        for (unsigned int boneID=0; boneID<humanModel->header.numberOfBones; boneID++)
        {
         if ( (humanModel->bones[boneID].info!=0) && (humanModel->bones[boneID].boneName!=0) ) //humanPose->x =
         {
          //fprintf(stderr,GREEN "BoneID %u  \n" NORMAL,boneID);
          struct pose6D axisPose={0};
          axisPose.x = humanPose->x + humanModel->bones[boneID].info->x;
          axisPose.y = humanPose->y + humanModel->bones[boneID].info->y;
          axisPose.z = humanPose->z + humanModel->bones[boneID].info->z;

          axisPose.usePoseMatrixDirectly = 1;
          for (unsigned int i=0; i<16; i++)
          {
            axisPose.m.m[i] = humanModel->bones[boneID].info->finalVertexTransformation[i];
          }
          axisPose.m.m[3]  = 10 * axisPose.x; //X
          axisPose.m.m[7]  = 10 * axisPose.y; //Y
          axisPose.m.m[11] = 10 * axisPose.z; //Z

          //axisPose.m.m[3]  = 10 *axisPose.y; //X
          axisPose.m.m[7]  -= 10;  //Y
          axisPose.m.m[11] -= 100; //Z

          if (
               (strcmp("scene",humanModel->bones[boneID].boneName)!=0)   &&
               (strcmp("camera",humanModel->bones[boneID].boneName)!=0)  &&
               (strcmp("light",humanModel->bones[boneID].boneName)!=0)   &&
               (strcmp("testexp",humanModel->bones[boneID].boneName)!=0) &&
               (strcmp("test",humanModel->bones[boneID].boneName)!=0)
             )
          {
           //Only draw axis of parts of actual geometry
           doOGLSingleDrawing(
                              programID,
                              MVPMatrixID,
                              0,
                              &axisPose,
                              axisVAO,
                              axisTriangleCount,
                              axisElementBuffer,
                              WIDTH,
                              HEIGHT
                            );
          }
         } else
         {
           fprintf(stderr,RED "BoneID %u empty! \n" NORMAL,boneID);
         }
        }

        //We have accumulated all data on the framebuffer and will now draw it back..
        drawFramebufferToScreen(
                                programFrameBufferID,
                                quad_vertexbuffer,
                                renderedTexture,
                                texID,
                                timeID,
                                resolutionID,
                                WIDTH,HEIGHT
                               );

		// Swap buffers
        glx3_endRedraw();


      //---------------------------------------------------------------
      //------------------- Calculate Framerate -----------------------
      //---------------------------------------------------------------
       unsigned long now=GetTickCountMicrosecondsIK();//GetTickCountMilliseconds();
       unsigned long elapsedTime=now-lastRenderingTime;
       if (elapsedTime==0) { elapsedTime=1; }
       lastFramerate = (float) 1000000/(elapsedTime);
       lastRenderingTime = now;
      //---------------------------------------------------------------
        ++framesRendered;
      //---------------------------------------------------------------

      if (renderForever) { usleep(1); } // Cap framerate if looping here...
	} // Check if the ESC key was pressed or the window was closed
    while(renderForever);


	// Cleanup VBO and shader
	glDeleteBuffers(1, &quad_vertexbuffer);
	glDeleteBuffers(1, &axisArrayBuffer);
	glDeleteBuffers(1, &axisElementBuffer);
	glDeleteVertexArrays(1, &axisVAO);
	return 1;
}






int doBVHDraw(
               GLuint programID,
               GLuint programFrameBufferID,
               GLuint FramebufferName,
               GLuint renderedTexture,
               GLuint renderedDepth,
               struct pose6D * humanPose,
               struct BVH_MotionCapture * bvh,
               unsigned int frameID,
               struct TRI_Model * axisModel,
               unsigned int WIDTH,
               unsigned int HEIGHT,
               int renderForever
             )
{
 	// Get a handle for our "MVP" uniform
	GLuint MVPMatrixID = glGetUniformLocation(programID, "MVP");
    //------------------------------------------------------------------------------------
    GLuint axisVAO=0;
    GLuint axisArrayBuffer=0;
    GLuint axisElementBuffer=0;
    unsigned int axisTriangleCount=0;

    int usePrimitive = 0;

    if (!usePrimitive)
    {
     axisTriangleCount  =  (unsigned int)  axisModel->header.numberOfVertices/3;
     pushObjectToBufferData(
                             1,
                             &axisVAO,
                             &axisArrayBuffer,
                             &axisElementBuffer,
                             programID,
                             axisModel->vertices       ,  axisModel->header.numberOfVertices      * sizeof(float),
                             axisModel->normal         ,  axisModel->header.numberOfNormals       * sizeof(float),
                             axisModel->textureCoords  ,  axisModel->header.numberOfTextureCoords * sizeof(float),      //0,0 //No Texture
                             axisModel->colors         ,  axisModel->header.numberOfColors        * sizeof(float),
                             axisModel->indices        ,  axisModel->header.numberOfIndices       * sizeof(unsigned int)//0,0 //Not Indexed
                           );
    } else
    {
     axisTriangleCount  = pyramidTriangleCount;
     pushObjectToBufferData(
                             1,
                             &axisVAO,
                             &axisArrayBuffer,
                             &axisElementBuffer,
                             programID,
                             pyramidCoords     ,  sizeof(pyramidCoords),
                             pyramidNormals    ,  sizeof(pyramidNormals),
                             pyramidTexCoords  ,  sizeof(pyramidTexCoords),      //0,0 //No Texture
                             cubeColors        ,  sizeof(cubeColors),
                             0        ,  0//0,0 //Not Indexed
                          );
    }
    //------------------------------------------------------------------------------------
	 GLuint quad_vertexbuffer;
	 glGenBuffers(1, &quad_vertexbuffer);
	 glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
	 glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);

	 // Create and compile our GLSL program from the shaders
	 GLuint texID = glGetUniformLocation(programFrameBufferID, "renderedTexture");
	 // Create and compile our GLSL program from the shaders
	 GLuint timeID = glGetUniformLocation(programFrameBufferID, "iTime");

	 GLuint resolutionID = glGetUniformLocation(programFrameBufferID, "iResolution");

	 do
     {
        // Render to our framebuffer
		glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
		glViewport(0,0,WIDTH,HEIGHT); // Render on the whole framebuffer, complete from the lower left corner to the upper right

       //-----------------------------------------------
        if (framesRendered%10==0) { fprintf(stderr,"\r%0.2f FPS                                         \r", lastFramerate ); }
       //-----------------------------------------------

	    glClearColor(backgroundColor[0],backgroundColor[1],backgroundColor[2],1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 		// Clear the screen


        struct BVH_Transform bvhTransform= {0};
        if (
                bvh_loadTransformForFrame(
                                           bvh,
                                           frameID,
                                           &bvhTransform,
                                           0
                                         )
             )
          {
           for (BVHJointID jID=0; jID<bvh->jointHierarchySize; jID++)
            {
               struct pose6D axisPose={0};
               axisPose.x = bvhTransform.joint[jID].pos3D[0]/10;
               axisPose.y = bvhTransform.joint[jID].pos3D[1]/10;
               axisPose.z = 50+bvhTransform.joint[jID].pos3D[2]/10;

               /*
                 struct Matrix4x4OfFloats localToWorldTransformation;
                 struct Matrix4x4OfFloats chainTransformation;
                 struct Matrix4x4OfFloats dynamicTranslation;
                 struct Matrix4x4OfFloats dynamicRotation; */
               axisPose.usePoseMatrixDirectly = 1;
               for (unsigned int i=0; i<16; i++)
                 {
                   axisPose.m.m[i] = bvhTransform.joint[jID].localToWorldTransformation.m[i];
                 }
               axisPose.m.m[3]  = axisPose.m.m[3]/10;
               axisPose.m.m[7]  = axisPose.m.m[7]/10;
               axisPose.m.m[11] = 50 + axisPose.m.m[11]/10 ; //Send rendering 50 units away..

               doOGLSingleDrawing(
                                   programID,
                                   MVPMatrixID,
                                   0,
                                   &axisPose,
                                   axisVAO,
                                   axisTriangleCount,
                                   axisElementBuffer,
                                   WIDTH,
                                   HEIGHT
                                 );
            }
          }

        //We have accumulated all data on the framebuffer and will now draw it back..
        drawFramebufferToScreen(
                                programFrameBufferID,
                                quad_vertexbuffer,
                                renderedTexture,
                                texID,
                                timeID,
                                resolutionID,
                                WIDTH,HEIGHT
                               );

		// Swap buffers
        glx3_endRedraw();


      //---------------------------------------------------------------
      //------------------- Calculate Framerate -----------------------
      //---------------------------------------------------------------
       unsigned long now=GetTickCountMicrosecondsIK();//GetTickCountMilliseconds();
       unsigned long elapsedTime=now-lastRenderingTime;
       if (elapsedTime==0) { elapsedTime=1; }
       lastFramerate = (float) 1000000/(elapsedTime);
       lastRenderingTime = now;
      //---------------------------------------------------------------
        ++framesRendered;
      //---------------------------------------------------------------

      if (renderForever) { usleep(1); } // Cap framerate if looping here...
	} // Check if the ESC key was pressed or the window was closed
    while(renderForever);


	// Cleanup VBO and shader
	glDeleteBuffers(1, &quad_vertexbuffer);
	glDeleteBuffers(1, &axisArrayBuffer);
	glDeleteBuffers(1, &axisElementBuffer);
	glDeleteVertexArrays(1, &axisVAO);
	return 1;
}




 /*
 location&rotation
 oculi01.l/r
 risorius03.l/r
 levator06.l/r
 oris03.l/r
 oris05
 levator05.l/r
 oris07.l/r
 oris01

 rotation
 orbicularis03.l/r
 orbicularis04.l/r
 eye.l/r */
void randomizeHead(struct BVH_MotionCapture * mc)
{
   BVHJointID jIDLeft;
   BVHJointID jIDRight;
   float r = 0.0;

   //===========================================================================
   if (
        (bvh_getJointIDFromJointNameNocase(mc,"neck",&jIDLeft))
      )
    {
      r = randomFloatA(-60,60);
      bvh_setJointRotationZAtFrame(mc,jIDLeft,0,r);
      r = randomFloatA(-30,30);
      bvh_setJointRotationXAtFrame(mc,jIDLeft,0,r);
      r = randomFloatA(-30,30);
      bvh_setJointRotationYAtFrame(mc,jIDLeft,0,r);
    }


   //                            ROTATION ONLY
   //===========================================================================
   //===========================================================================
   //===========================================================================
   if (
        (bvh_getJointIDFromJointNameNocase(mc,"orbicularis03.l",&jIDLeft)) &&
        (bvh_getJointIDFromJointNameNocase(mc,"orbicularis03.r",&jIDRight))
      )
    {
      r = randomFloatA(-30,30);
      bvh_setJointRotationZAtFrame(mc,jIDLeft,0,r);
      bvh_setJointRotationZAtFrame(mc,jIDRight,0,r);
    }
   //===========================================================================
   //===========================================================================
   //===========================================================================
   if (
        (bvh_getJointIDFromJointNameNocase(mc,"orbicularis04.l",&jIDLeft)) &&
        (bvh_getJointIDFromJointNameNocase(mc,"orbicularis04.r",&jIDRight))
      )
    {
      r = randomFloatA(-30,30);
      bvh_setJointRotationZAtFrame(mc,jIDLeft,0,r);
      bvh_setJointRotationZAtFrame(mc,jIDRight,0,r);
    }
   //===========================================================================
   //===========================================================================
   //===========================================================================
   if (
        (bvh_getJointIDFromJointNameNocase(mc,"eye.l",&jIDLeft)) &&
        (bvh_getJointIDFromJointNameNocase(mc,"eye.r",&jIDRight))
      )
    {
      r = randomFloatA(-30,30);
      bvh_setJointRotationZAtFrame(mc,jIDLeft,0,r);
      bvh_setJointRotationZAtFrame(mc,jIDRight,0,r);

      r = randomFloatA(-30,30);
      bvh_setJointRotationXAtFrame(mc,jIDLeft,0,r);
      bvh_setJointRotationXAtFrame(mc,jIDRight,0,r);
    }




   //                         LOCATION &  ROTATION ONLY
   //===========================================================================
   //===========================================================================
   //===========================================================================
   if (
        (bvh_getJointIDFromJointNameNocase(mc,"oculi01.l",&jIDLeft)) &&
        (bvh_getJointIDFromJointNameNocase(mc,"oculi01.r",&jIDRight))
      )
    {
      r = randomFloatA(-30,30);
      bvh_setJointRotationZAtFrame(mc,jIDLeft,0,r);
      bvh_setJointRotationZAtFrame(mc,jIDRight,0,r);
      r = randomFloatA(-30,30);
      bvh_setJointRotationYAtFrame(mc,jIDLeft,0,r);
      bvh_setJointRotationYAtFrame(mc,jIDRight,0,r);
    }
   //===========================================================================
   //===========================================================================
   //===========================================================================
   if (
        (bvh_getJointIDFromJointNameNocase(mc,"risorius03.l",&jIDLeft)) &&
        (bvh_getJointIDFromJointNameNocase(mc,"risorius03.r",&jIDRight))
      )
    {
      r = randomFloatA(-50,50);
      bvh_setJointRotationZAtFrame(mc,jIDLeft,0,r);
      bvh_setJointRotationZAtFrame(mc,jIDRight,0,r);
    }
   //===========================================================================
   //===========================================================================
   //===========================================================================
   if (
        (bvh_getJointIDFromJointNameNocase(mc,"levator06.l",&jIDLeft)) &&
        (bvh_getJointIDFromJointNameNocase(mc,"levator06.r",&jIDRight))
      )
    {
      r = randomFloatA(-20,20);
      bvh_setJointRotationXAtFrame(mc,jIDLeft,0,r);
      bvh_setJointRotationXAtFrame(mc,jIDRight,0,r);
      r = randomFloatA(-10,30);
      bvh_setJointRotationZAtFrame(mc,jIDLeft,0,-r);
      bvh_setJointRotationZAtFrame(mc,jIDRight,0,r);
    }
   //===========================================================================
   //===========================================================================
   //===========================================================================
   if (
        (bvh_getJointIDFromJointNameNocase(mc,"oris03.l",&jIDLeft)) &&
        (bvh_getJointIDFromJointNameNocase(mc,"oris03.r",&jIDRight))
      )
    {
      r = randomFloatA(-30,30);
      bvh_setJointRotationYAtFrame(mc,jIDLeft,0,r);
      bvh_setJointRotationYAtFrame(mc,jIDRight,0,r);
      r = randomFloatA(-30,30);
      bvh_setJointRotationXAtFrame(mc,jIDLeft,0,r);
      bvh_setJointRotationXAtFrame(mc,jIDRight,0,r);
    }
   //===========================================================================
   //===========================================================================
   //===========================================================================
   if (
        (bvh_getJointIDFromJointNameNocase(mc,"oris05",&jIDLeft))
      )
    {
      r = randomFloatA(-30,30);
      bvh_setJointRotationYAtFrame(mc,jIDLeft,0,r);
      r = randomFloatA(-30,30);
      bvh_setJointRotationXAtFrame(mc,jIDLeft,0,r);
    }
   //===========================================================================
   //===========================================================================
   //===========================================================================
   if (
        (bvh_getJointIDFromJointNameNocase(mc,"oris01",&jIDLeft))
      )
    {
      r = randomFloatA(-30,30);
      bvh_setJointRotationYAtFrame(mc,jIDLeft,0,r);
      r = randomFloatA(-30,30);
      bvh_setJointRotationXAtFrame(mc,jIDLeft,0,r);
    }
   //===========================================================================
   //===========================================================================
   //===========================================================================
   if (
        (bvh_getJointIDFromJointNameNocase(mc,"levator05.l",&jIDLeft)) &&
        (bvh_getJointIDFromJointNameNocase(mc,"levator05.r",&jIDRight))
      )
    {
      r = randomFloatA(-60,60);
      bvh_setJointRotationYAtFrame(mc,jIDLeft,0,r);
      bvh_setJointRotationYAtFrame(mc,jIDRight,0,r);
    }
   //===========================================================================
   //===========================================================================
   //===========================================================================
   if (
        (bvh_getJointIDFromJointNameNocase(mc,"oris07.l",&jIDLeft)) &&
        (bvh_getJointIDFromJointNameNocase(mc,"oris07.r",&jIDRight))
      )
    {
      r = randomFloatA(-30,30);
      bvh_setJointRotationZAtFrame(mc,jIDLeft,0,r);
      bvh_setJointRotationZAtFrame(mc,jIDRight,0,r);
    }
}


int getAll2DEncodedPoints(unsigned char * pixels,unsigned int width,unsigned int height,unsigned int numberOfUniqueColors)
{
    if (pixels==0) { fprintf(stderr,RED "getAll2DEncodedPoints no pixels \n" NORMAL); return 0; }
    fprintf(stderr,CYAN "getAll2DEncodedPoints \n" NORMAL);

    unsigned int * m = (unsigned int  *) malloc(sizeof(unsigned int ) * numberOfUniqueColors * 2);

    if (m!=0)
    {
    // ./gl3MeshTransform --randomize --set hip x 0 --face --texture occupiedTexture.pnm --eyetexture occupiedEyesTexture.pnm --noeyehair --dump2D face.txt
    unsigned char * imageEnd = pixels + ( width * height * 3);
    unsigned char * ptr = pixels;
    while (ptr<imageEnd-3)
    {
        unsigned char r = *ptr; ++ptr;
        unsigned char g = *ptr; ++ptr;
        unsigned char b = *ptr; ++ptr;

        if ( (r!=0) || (g!=0) || (b!=0) )//CAREFUL RENDERED PIXELS DONT TAKE EXACT COLOR
        {
        // DO Decoding here..!
          unsigned int doubleCheckedPoint = decodeUniqueColor(numberOfUniqueColors,r,g,b);

          unsigned int pixelsTraversed = (unsigned int) (ptr - pixels) / 3;
          unsigned int y = (unsigned int) pixelsTraversed / width;
          unsigned int x = (unsigned int) pixelsTraversed % width;
          fprintf(stdout,"POINT(%u,%u=%u - %u)\n",x,y,doubleCheckedPoint,numberOfUniqueColors);

          m[doubleCheckedPoint*2+0] = x;
          m[doubleCheckedPoint*2+1] = y;

          //Direct log to a file..!
          FILE *fp = fopen("outpoints.dat","a");
          if (fp!=0)
          {
            fprintf(fp,"POINT(%u,%u=%u)\n",x,y,doubleCheckedPoint);
            fclose(fp);
          }
          //exit(0);
          //return 1;
        }
    }


    free(m);
    }

    return 0;
}



int getTextureActivation(unsigned char * pixels,unsigned int width,unsigned int height,unsigned int flashX,unsigned int flashY)
{
    unsigned char * imageEnd = pixels + ( width * height * 3);
    unsigned char * ptr = pixels;
    while (ptr<imageEnd-3)
    {
        unsigned char r = *ptr; ++ptr;
        unsigned char g = *ptr; ++ptr;
        unsigned char b = *ptr; ++ptr;

        if ( (r!=0) || (g!=0) || (b!=0) )//CAREFUL RENDERED PIXELS DONT TAKE EXACT COLOR
        {
          unsigned int pixelsTraversed = (unsigned int) (ptr - pixels) / 3;
          unsigned int y = (unsigned int) pixelsTraversed / width;
          unsigned int x = (unsigned int) pixelsTraversed % width;
          fprintf(stdout,"HIT(%u,%u,%u,%u)\n",x,y,flashX,flashY);

          //Direct log to a file..!
          FILE *fp = fopen("textureActivation.dat","a");
          if (fp!=0)
          {
            fprintf(fp,"HIT(%u,%u,%u,%u)\n",x,y,flashX,flashY);
            fclose(fp);
          }
          //exit(0);
          return 1;
        }

    }

    return 0;
}


int setTexturePixel(GLuint programID,struct TRI_Model * model, unsigned int x,unsigned int y)
{
      memset(
             model->textureData,
             0, //rand()%255,
             sizeof(char) * model->header.textureDataWidth * model->header.textureDataHeight * model->header.textureDataChannels
            );

    unsigned char aR = flashR;
    unsigned char aG = flashG;
    unsigned char aB = flashB;

    fprintf(stderr,"X=%u,Y=%u ",x,y);
    unsigned char * ptr = model->textureData + (y * model->header.textureDataWidth * model->header.textureDataChannels) + (x * model->header.textureDataChannels);

    unsigned char * r = ptr; ptr++;
    unsigned char * g = ptr; ptr++;
    unsigned char * b = ptr; ptr++;
    *r = aR;
    *g = aG;
    *b = aB;

    uploadColorImageAsTexture(
                               programID,
                               (GLuint *) &model->header.textureBindGLBuffer,
                               &model->header.textureUploadedToGPU,
                               (unsigned char*) model->textureData,
                               model->header.textureDataWidth,
                               model->header.textureDataHeight,
                               model->header.textureDataChannels,
                               24
                             );
}



int main(int argc,const char **argv)
{
  //Disable VSYNC
  if (VSYNC==0)
  {
   disableVSync();
  }


  unsigned int WIDTH =(unsigned int) (tilesToDoX*originalWIDTH)/shrinkingFactor;
  unsigned int HEIGHT=(unsigned int) (tilesToDoY*originalHEIGHT)/shrinkingFactor;


  int visibleWindow = 1;
  for (int i=0; i<argc; i++)
        {
             if (strcmp(argv[i],"--flashtexture")==0)
                    {
                       //When Doing a batch processing job like flashing texture use an invisible
                       //window to make sure renderings happen regardless of window visibility
		               fprintf(stderr, "Using invisible window\n");
                       visibleWindow = 0;
                    } else
             if (strcmp(argv[i],"--dump2D")==0)
                    {
                       //When Doing a batch processing job like flashing texture use an invisible
                       //window to make sure renderings happen regardless of window visibility
		               fprintf(stderr, "Using invisible window\n");
                       //visibleWindow = 0;
                    }


        }






  fprintf(stderr,"Attempting to setup a %ux%u glx3 context\n",WIDTH,HEIGHT);
  start_glx3_stuff(WIDTH,HEIGHT,visibleWindow,argc,argv);

  //If you dont want VSYNC run with
  //vblank_mode=0 __GL_SYNC_TO_VBLANK=0 ./gl3MeshTransform
  if (glewInit() != GLEW_OK)
   {
		fprintf(stderr, "Failed to initialize GLEW\n");
	 	return 1;
   }

  GLuint programID=0;
  GLuint programFrameBufferID=0;
  GLuint FramebufferName=0;
  GLuint renderedTexture=0;
  GLuint renderedDepth=0;

  int skinnedRendering=(performBoneTransformsInCPU==0); //<--auto set

  if (!initializeOGLRenderer(&programID,&programFrameBufferID,&FramebufferName,&renderedTexture,&renderedDepth,skinnedRendering,WIDTH,HEIGHT))
  {
		fprintf(stderr, "Failed to initialize Shaders\n");
	 	return 1;
  }

   #define defaulBVHToLoad "01_02.bvh" //merged_neutral.bvh
   #define defaultModelToLoad "makehuman.tri"

   const char * bvhToLoad = defaulBVHToLoad;
   const char * modelToLoad = defaultModelToLoad;

   //------------------------------------------------------
   struct BVH_MotionCapture mc = {0};
   //------------------------------------------------------
   struct TRI_Model axisModel={0};
   //------------------------------------------------------
   struct TRI_Model eyeModel={0};
   struct TRI_Model indexedEyeModel={0};
   //------------------------------------------------------
   struct TRI_Model hairModel={0};
   struct TRI_Model indexedHairModel={0};
   //------------------------------------------------------
   struct TRI_Model eyebrowsModel={0};
   struct TRI_Model indexedEyebrowsModel={0};
   //------------------------------------------------------
   struct TRI_Model eyelashesModel={0};
   struct TRI_Model indexedEyelashesModel={0};
   //------------------------------------------------------
   struct pose6D humanPose={0};
   struct TRI_Model humanModel={0};
   struct TRI_Model indexedHumanModel={0};
   //------------------------------------------------------
   int destroyColors=0;
   int dumpVideo = 0;
   int dumpSnapshot = 0;
   unsigned int maxFrames=0;

   int axisRendering = 0;
   int staticRendering = 0;

   int randomize=0;

   int limit = 0;
   /*
   //DAE output Set human pose to somewhere visible..
   //-------------------------------------------------------------------
   humanPose.roll=180.0;//(float)  (rand()%90);
   humanPose.pitch=0.0;//(float) (rand()%90);
   humanPose.yaw=0.0;//(float)   (rand()%90);
   //-------------------------------------------------------------------
   humanPose.x=0.0f;//(float)  (1000-rand()%2000);
   humanPose.y=-7.976f;//(float) (100-rand()%200);
   humanPose.z=27.99735f;//(float)  (700+rand()%1000);
   //-------------------------------------------------------------------
*/

   //MHX2 Set human pose to somewhere visible..
   //-------------------------------------------------------------------
   humanPose.roll=0.0;//180.0;//(float)  (rand()%90);
   humanPose.pitch=90.0;//-90.0;//(float) (rand()%90);
   humanPose.yaw=0.0;//(float)   (rand()%90);
   //-------------------------------------------------------------------
   humanPose.x=0.0f;//(float)  (1000-rand()%2000);
   humanPose.y=-0.976f;//(float) (100-rand()%200);
   humanPose.z=2.99735f;//(float)  (700+rand()%1000);
   //-------------------------------------------------------------------

   //------------------------------------------------------
   for (int i=0; i<argc; i++)
        {
           if (strcmp(argv[i],"--limit")==0)
                    {
                      limit=atoi(argv[i+1]);
                    } else
           if (strcmp(argv[i],"--axis")==0)
                    {
                      axisRendering=1;
                    } else
           if (strcmp(argv[i],"--bvhaxis")==0)
                    {
                      axisRendering=2;
                    } else
           if (strcmp(argv[i],"--static")==0)
                    {
                      staticRendering=1;
                    } else
           if (strcmp(argv[i],"--face")==0)
                    {
                       //  ./gl3MeshTransform --face --set eye.l x 20 --set eye.r x 20 --set eye.l z 20 --set eye.r z 20 --set orbicularis03.l x 30 --set orbicularis03.r x 30
                       //  ./gl3MeshTransform --face --set eye.l x 20 --set eye.r x 20 --set eye.l z 20 --set eye.r z 20
                       //  ./gl3MeshTransform --face --bvh merged_neutral.bvh

                       //Regular DAE align global=local
                       humanPose.x=0.0f;//(float)  (1000-rand()%2000);
                       humanPose.y=-14.976f;//(float) (100-rand()%200);
                       humanPose.z=7.99735f;//(float)  (700+rand()%1000);

                       //MHX2
                       humanPose.x=0.0f;//(float)  (1000-rand()%2000);
                       humanPose.y=-1.476f;//(float) (100-rand()%200);
                       humanPose.z=0.69735f;//(float)  (700+rand()%1000);
                    } else
           if (strcmp(argv[i],"--dump2D")==0)
                    {
                      // ./gl3MeshTransform --randomize --set hip x 0 --face --texture occupiedTexture.pnm --eyetexture occupiedEyesTexture.pnm --noeyehair --dump2D face.txt
                      dump2DPointOutput=1;

                      unsigned int numberOfPoints = 0;
                      unsigned int  * keypoints = readKeyPoint(argv[i+1],1,originalWIDTH,originalHEIGHT,&numberOfPoints);
                      if (keypoints==0)
                      {
                          fprintf(stderr,"Cannot read key points %s",argv[i+1]);
                          return 0;
                      }
                      numberOfUniqueColors = numberOfPoints + 2; // We also allocated 2 more points for l/r eyes!
                      if (keypoints!=0) { free(keypoints);}

                      FILE *fp = fopen("outpoints.dat","w");
                      if (fp!=0)
                          {
                           fclose(fp);
                          }
                    } else
           if (strcmp(argv[i],"--flashtexture")==0)
                    {
                        // ./gl3MeshTransform --set hip x 0 --face --flashtexture 2>/dev/null
                        flashTexturePixels = 1;
                        renderEyes = 0;
                        FILE *fp = fopen("textureActivation.dat","w");
                        if (fp!=0)
                          {
                           fclose(fp);
                          }
                    } else
           if (strcmp(argv[i],"--save")==0)
                    {
                      dumpSnapshot=1;
                    } else
           if (strcmp(argv[i],"--dumpvideo")==0)
                    {
                      dumpVideo=1;
                    } else
           if (strcmp(argv[i],"--nocolor")==0)
                    {
                      destroyColors=1;
                    } else
           if (strcmp(argv[i],"--noeyehair")==0)
                    {
                      renderEyeHair=0;
                    } else
           if (strcmp(argv[i],"--bvh")==0)
                    {
                        if (argc>i+1)
                            {
                                bvhToLoad = argv[i+1];
                            }
                    } else
           if (strcmp(argv[i],"--from")==0)
                    {
                        if (argc>i+1)
                            {
                                modelToLoad = argv[i+1];
                            }
                    } else
           if (strcmp(argv[i],"--maxFrames")==0)
                    {
                        if (argc>i+1)
                            {
                                maxFrames = atoi(argv[i+1]);
                            }
                    }
        }
   //------------------------------------------------------
   if (!bvh_loadBVH(bvhToLoad,&mc,1.0) ) // This is the new armature that includes the head
        {
          fprintf(stderr,"Cannot find the merged_neutral.bvh file..\n");
          return 0;
        }
   bvh_renameJointsToLowercase(&mc);
   //------------------------------------------------------
   if (axisRendering)
   {
     struct TRI_Model axisModelIndexed={0};
     if (!tri_loadModel("axis.tri", &axisModelIndexed ) )
     {
       fprintf(stderr,"Please : wget http://ammar.gr/mocapnet/axis.tri\n");
       return 0;
     }
     //paintTRI(&axisModelIndexed,123,123,123);
     tri_flattenIndexedModel(&axisModel,&axisModelIndexed);
   }
   //------------------------------------------------------
   if (!tri_loadModel(modelToLoad, &indexedHumanModel ) )
   {
     fprintf(stderr,"Please : wget http://ammar.gr/mocapnet/makehuman.tri\n");
     return 0;
   }
   //------------------------------------------------------
   if (!tri_loadModel("eyes.tri", &indexedEyeModel ) )
   {
     fprintf(stderr,"Please : wget http://ammar.gr/mocapnet/eyes.tri\n");
     return 0;
   }
   //------------------------------------------------------
   if (!tri_loadModel("hair.tri", &indexedHairModel ) )
   {
     fprintf(stderr,"Please : wget http://ammar.gr/mocapnet/hair.tri\n");
     return 0;
   }
   //------------------------------------------------------
   if (!tri_loadModel("eyebrows.tri", &indexedEyebrowsModel ) )
   {
     fprintf(stderr,"Please : wget http://ammar.gr/mocapnet/eyebrows.tri\n");
     return 0;
   }
   //------------------------------------------------------
   if (!tri_loadModel("eyelashes.tri", &indexedEyelashesModel ) )
   {
     fprintf(stderr,"Please : wget http://ammar.gr/mocapnet/eyelashes.tri\n");
     return 0;
   }

   if (destroyColors)
   {
      tri_paintModel(&indexedHumanModel,123,123,123);
      tri_paintModel(&indexedEyeModel,123,123,123);
      tri_paintModel(&indexedHairModel,123,123,123);
      tri_paintModel(&indexedEyebrowsModel,123,123,123);
      tri_paintModel(&indexedEyelashesModel,123,123,123);
   }
   //------------------------------------------------------

   unsigned char * rgb = 0;
   if ( (dumpVideo) || (dumpSnapshot) || (flashTexturePixels) || (dump2DPointOutput) )
      { rgb =  (unsigned char * ) malloc(sizeof(unsigned char) * WIDTH * HEIGHT *3); }

   //------------------------------------------------------
   fprintf(stderr,"Preprocessing human model.. ");
   tri_makeAllBoneNamesLowerCase(&indexedHumanModel);
   tri_removePrefixFromAllBoneNames(&indexedHumanModel,"test_"); //Eyes have a test_ prefix on bone names..
   tri_removePrefixFromAllBoneNames(&indexedHumanModel,"f_"); //Fingers have a f_ prefix on bone names..
   makeAllTRIBoneNamesLowerCaseWithoutUnderscore(&indexedHumanModel);
   fprintf(stderr,GREEN "OK\n" NORMAL);
   //------------------------------------------------------
   fprintf(stderr,"Preprocessing eye model.. ");
   tri_makeAllBoneNamesLowerCase(&indexedEyeModel);
   tri_removePrefixFromAllBoneNames(&indexedEyeModel,"test_"); //Eyes have a test_ prefix on bone names..
   makeAllTRIBoneNamesLowerCaseWithoutUnderscore(&indexedEyeModel);
   fprintf(stderr,GREEN "OK\n" NORMAL);
   //------------------------------------------------------
   fprintf(stderr,"Preprocessing hair model.. ");
   tri_makeAllBoneNamesLowerCase(&indexedHairModel);
   tri_removePrefixFromAllBoneNames(&indexedHairModel,"test_"); //Eyes have a test_ prefix on bone names..
   makeAllTRIBoneNamesLowerCaseWithoutUnderscore(&indexedHairModel);
   fprintf(stderr,GREEN "OK\n" NORMAL);
   //------------------------------------------------------
   fprintf(stderr,"Preprocessing eyebrows model.. ");
   tri_makeAllBoneNamesLowerCase(&indexedEyebrowsModel);
   tri_removePrefixFromAllBoneNames(&indexedEyebrowsModel,"test_"); //Eyes have a test_ prefix on bone names..
   makeAllTRIBoneNamesLowerCaseWithoutUnderscore(&indexedEyebrowsModel);
   fprintf(stderr,GREEN "OK\n" NORMAL);
   //------------------------------------------------------
   fprintf(stderr,"Preprocessing eyelashes model.. ");
   tri_makeAllBoneNamesLowerCase(&indexedEyelashesModel);
   tri_removePrefixFromAllBoneNames(&indexedEyelashesModel,"test_"); //Eyes have a test_ prefix on bone names..
   makeAllTRIBoneNamesLowerCaseWithoutUnderscore(&indexedEyelashesModel);
   fprintf(stderr,GREEN "OK\n" NORMAL);
   //------------------------------------------------------

   bvh_printBVH(&mc);
   printTRIBoneStructure(&indexedHumanModel,0 /*alsoPrintMatrices*/);



     for (int i=0; i<argc; i++)
        {
           if (strcmp(argv[i],"--hair")==0)
                    {
                      renderHair = 1;
                    } else
           if (strcmp(argv[i],"--parse")==0)
                    {
                      //  ./gl3MeshTransform --parse textureActivation.dat face.txt
                      parseTextureToScreenAssociations(argv[i+1],argv[i+2],&indexedHumanModel,&indexedEyeModel);
                      exit(0);
                    } else
           if (strcmp(argv[i],"--colorcode")==0)
                    {
                        // 2000 1400
                        tri_colorCodeTexture(&indexedHumanModel,1600,700,400,700);
                    } else
           if (strcmp(argv[i],"--texture")==0)
                    {
                        struct Image newImg={0};
                        ReadPPM(argv[i+1],&newImg,0);
                        if ( indexedHumanModel.textureData!=0 )
                        {
                            free(indexedHumanModel.textureData);
                            indexedHumanModel.textureData = 0;
                        }

                        indexedHumanModel.textureData = newImg.pixels;
                        indexedHumanModel.header.textureDataWidth = newImg.width;
                        indexedHumanModel.header.textureDataHeight = newImg.height;
                        indexedHumanModel.header.textureDataChannels = newImg.channels;
                    } else
           if (strcmp(argv[i],"--eyetexture")==0)
                    {
                        //./gl3MeshTransform --randomize --set hip x 0 --face --texture occupiedTexture.pnm --eyetexture occupiedEyesTexture.pnm
                        struct Image newImg={0};
                        ReadPPM(argv[i+1],&newImg,0);
                        if ( indexedEyeModel.textureData!=0 )
                        {
                            free(indexedEyeModel.textureData);
                            indexedEyeModel.textureData = 0;
                        }

                        indexedEyeModel.textureData = newImg.pixels;
                        indexedEyeModel.header.textureDataWidth = newImg.width;
                        indexedEyeModel.header.textureDataHeight = newImg.height;
                        indexedEyeModel.header.textureDataChannels = newImg.channels;
                    }
        }



   if (indexedHumanModel.textureData!=0)
   {
      uploadColorImageAsTexture(
                                 programID,
                                 (GLuint *) &indexedHumanModel.header.textureBindGLBuffer,
                                 &indexedHumanModel.header.textureUploadedToGPU,
                                 (unsigned char*) indexedHumanModel.textureData,
                                  indexedHumanModel.header.textureDataWidth,
                                  indexedHumanModel.header.textureDataHeight,
                                  indexedHumanModel.header.textureDataChannels,
                                  24
                                );
      //Don't copy data around since its already on GPU
      //free(indexedHumanModel.textureData);
      //indexedHumanModel.textureData=0;
   }

   if (indexedEyeModel.textureData!=0)
   {
      uploadColorImageAsTexture(
                                 programID,
                                 (GLuint *) &indexedEyeModel.header.textureBindGLBuffer,
                                 &indexedEyeModel.header.textureUploadedToGPU,
                                 (unsigned char*) indexedEyeModel.textureData,
                                  indexedEyeModel.header.textureDataWidth,
                                  indexedEyeModel.header.textureDataHeight,
                                  indexedEyeModel.header.textureDataChannels,
                                  24
                                );
      //Don't copy data around since its already on GPU
      //free(indexedEyeModel.textureData);
      //indexedEyeModel.textureData=0;
   }

   if (indexedHairModel.textureData!=0)
   {
      uploadColorImageAsTexture(
                                 programID,
                                 (GLuint *) &indexedHairModel.header.textureBindGLBuffer,
                                 &indexedHairModel.header.textureUploadedToGPU,
                                 (unsigned char*) indexedHairModel.textureData,
                                  indexedHairModel.header.textureDataWidth,
                                  indexedHairModel.header.textureDataHeight,
                                  indexedHairModel.header.textureDataChannels,
                                  24
                                );
      //Don't copy data around since its already on GPU
      //free(indexedHairModel.textureData);
      //indexedHairModel.textureData=0;
   }

   if (indexedEyebrowsModel.textureData!=0)
   {
      uploadColorImageAsTexture(
                                 programID,
                                 (GLuint *) &indexedEyebrowsModel.header.textureBindGLBuffer,
                                 &indexedEyebrowsModel.header.textureUploadedToGPU,
                                 (unsigned char*) indexedEyebrowsModel.textureData,
                                  indexedEyebrowsModel.header.textureDataWidth,
                                  indexedEyebrowsModel.header.textureDataHeight,
                                  indexedEyebrowsModel.header.textureDataChannels,
                                  24
                                );
      //Don't copy data around since its already on GPU
      //free(indexedEyebrowsModel.textureData);
      //indexedEyebrowsModel.textureData=0;
   }


   if (indexedEyelashesModel.textureData!=0)
   {
      uploadColorImageAsTexture(
                                 programID,
                                 (GLuint *) &indexedEyelashesModel.header.textureBindGLBuffer,
                                 &indexedEyelashesModel.header.textureUploadedToGPU,
                                 (unsigned char*) indexedEyelashesModel.textureData,
                                  indexedEyelashesModel.header.textureDataWidth,
                                  indexedEyelashesModel.header.textureDataHeight,
                                  indexedEyelashesModel.header.textureDataChannels,
                                  24
                                );
      //Don't copy data around since its already on GPU
      //free(indexedEyelashesModel.textureData);
      //indexedEyelashesModel.textureData=0;
   }


  //Pass models to the GPU/Shader structure that will consume them..
  //-------------------------------------------------------------------------------
  struct GPUTriModel gpuEyelashes={0}; gpuEyelashes.model = &indexedEyelashesModel;  gpuEyelashes.programID=programID;
  struct GPUTriModel gpuEyebrows={0};  gpuEyebrows.model  = &indexedEyebrowsModel;   gpuEyebrows.programID=programID;
  struct GPUTriModel gpuHair={0};      gpuHair.model      = &indexedHairModel;       gpuHair.programID=programID;
  struct GPUTriModel gpuEyes={0};      gpuEyes.model      = &indexedEyeModel;        gpuEyes.programID=programID;
  struct GPUTriModel gpuHuman={0};     gpuHuman.model     = &indexedHumanModel;      gpuHuman.programID=programID;
  //-------------------------------------------------------------------------------


  fprintf(stderr,"eyelashesTextureID = %u \n",indexedEyelashesModel.header.textureBindGLBuffer);
  fprintf(stderr,"eyebrowsTextureID = %u \n",indexedEyebrowsModel.header.textureBindGLBuffer);
  fprintf(stderr,"hairTextureID = %u \n",indexedHairModel.header.textureBindGLBuffer);
  fprintf(stderr,"eyeTextureID = %u \n",indexedEyeModel.header.textureBindGLBuffer);
  fprintf(stderr,"humanTextureID = %u \n",indexedHumanModel.header.textureBindGLBuffer);

  unsigned int startFlashX = 1600;
  unsigned int startFlashY = 750;
  unsigned int endFlashX = startFlashX + 400;
  unsigned int endFlashY = startFlashY + 630;
  unsigned int flashX=startFlashX;
  unsigned int flashY=startFlashY;


   //------------------------------------------------------
   //  ./gl3MeshTransform --set relbow z 90 --set hip y 180 --set lelbow x 90 --set rknee z 45 --set lshoulder y -45
   //  ./gl3MeshTransform --bvhaxis --set relbow z 90 --set hip y 180 --set lelbow y 90 --set rknee z -45 --set lshoulder x 45
   for (int i=0; i<argc; i++)
        {
           if (strcmp(argv[i],"--randomize")==0)
                    {
                       randomize=1;
                       mc.numberOfFrames=1;
                    } else
           if (strcmp(argv[i],"--set")==0)
                    {
                      const char * jointName = argv[i+1];
                      const char * jointAxis = argv[i+2];
                      float  jointValue = atof(argv[i+3]);
                      BVHJointID jID;

                      if (bvh_getJointIDFromJointNameNocase(&mc,jointName,&jID))
                      {
                          if (strcmp("x",jointAxis)==0)
                          {
                              bvh_setJointRotationXAtFrame(&mc,jID,0,jointValue);
                          } else
                          if (strcmp("y",jointAxis)==0)
                          {
                              bvh_setJointRotationYAtFrame(&mc,jID,0,jointValue);
                          } else
                          if (strcmp("z",jointAxis)==0)
                          {
                              bvh_setJointRotationZAtFrame(&mc,jID,0,jointValue);
                          }
                      }

                      mc.numberOfFrames=1;
                    }
        }
   //------------------------------------------------------

   //Test
   alignRotationOfTRIVsBVH(&indexedHumanModel,&mc,"lelbow","lelbow",0);
   alignRotationOfTRIVsBVH(&indexedHumanModel,&mc,"relbow","relbow",0);
   alignRotationOfTRIVsBVH(&indexedHumanModel,&mc,"lshoulder","lshoulder",1);
   alignRotationOfTRIVsBVH(&indexedHumanModel,&mc,"rshoulder","rshoulder",1);
   //exit(0);

   //We need to free this after application is done..
   unsigned int * humanMap = createLookupTableFromTRItoBVH(&indexedHumanModel,&mc,1);

   if (maxFrames==0)
   {
       maxFrames = mc.numberOfFrames;
   }

   do
    {
      if (randomize)
        {
            randomizeHead(&mc);
            usleep(10);
        }


    for (BVHFrameID fID=0; fID<maxFrames; fID++)
    {
     fprintf(stderr,CYAN "\nBVH %s Frame %u/%u (BVH has %u frames total) \n" NORMAL,mc.fileName,fID,maxFrames,mc.numberOfFrames);
     //-------------------------------------------


     if (flashTexturePixels)
     {
       flashX+=1;
       if (flashX>endFlashX)  {  flashX=startFlashX; flashY+=1;            }
       if (flashY>=endFlashY)  {  /*flashX=startFlashX; flashY=startFlashY;*/ exit(0); break; }
       else
       {
        //flashX = 1663; flashY = 1063; //HIT(559,423,1663,1063)
        fprintf(stderr,GREEN "%0.2f %% flashing pixels (%u,%u->%u,%u) \n\n" NORMAL,(float) 100*(flashY-startFlashY)/(endFlashY-startFlashY), flashX,flashY, endFlashX,endFlashY);
        setTexturePixel(programID,&indexedHumanModel,flashX,flashY);
       }
     }


     if (!staticRendering)
     {
       //We animate the model in CPU instead of the shader!
       //And just give the final calculated vertices for rendering
       animateTRIModelUsingBVHArmature(&humanModel    ,&indexedHumanModel    ,&mc,humanMap,fID,performBoneTransformsInCPU,0);
       animateTRIModelUsingBVHArmature(&eyeModel      ,&indexedEyeModel      ,&mc,humanMap,fID,performBoneTransformsInCPU,0);
       animateTRIModelUsingBVHArmature(&hairModel     ,&indexedHairModel     ,&mc,humanMap,fID,performBoneTransformsInCPU,0);
       animateTRIModelUsingBVHArmature(&eyebrowsModel ,&indexedEyebrowsModel ,&mc,humanMap,fID,performBoneTransformsInCPU,0);
       animateTRIModelUsingBVHArmature(&eyelashesModel,&indexedEyelashesModel,&mc,humanMap,fID,performBoneTransformsInCPU,0);

       if (performBoneTransformsInCPU)
       {
        gpuHuman.model     = &humanModel;
        gpuEyes.model      = &eyeModel;
        gpuHair.model      = &hairModel;
        gpuEyebrows.model  = &eyebrowsModel;
        gpuEyelashes.model = &eyelashesModel;
       } else
       {
        //TODO: When skinning takes place in shader we will just give the initial model..
        gpuEyelashes.model = &indexedEyelashesModel;
        gpuEyebrows.model  = &indexedEyebrowsModel;
        gpuHair.model      = &indexedHairModel;
        gpuEyes.model      = &indexedEyeModel;
        gpuHuman.model     = &indexedHumanModel;
       }
     } else
     {
       tri_flattenIndexedModel(&humanModel,&indexedHumanModel);
       tri_flattenIndexedModel(&eyeModel,&indexedEyeModel);
       tri_flattenIndexedModel(&hairModel,&indexedHairModel);
       tri_flattenIndexedModel(&eyebrowsModel,&indexedEyebrowsModel);
       tri_flattenIndexedModel(&eyelashesModel,&indexedEyelashesModel);
       gpuEyelashes.model = &eyelashesModel;
       gpuEyebrows.model  = &eyebrowsModel;
       gpuHair.model      = &hairModel;
       gpuEyes.model      = &eyeModel;
       gpuHuman.model     = &humanModel;
     }


    if (axisRendering == 1)
    {
      humanPose.roll+=1.0;//(float)  (rand()%90);
      humanPose.pitch+=1.0;//(float) (rand()%90);
      humanPose.yaw+=1.0;//(float)   (rand()%90);
      //-------------------------------------------------------------------
      humanPose.x=0.0f;//(float)  (1000-rand()%2000);
      humanPose.y=0.0f;//(float) (100-rand()%200);
      humanPose.z=13.4f;//(float)  (700+rand()%1000);



        struct BVH_Transform bvhTransform= {0};
        if (
                bvh_loadTransformForFrame(
                                           &mc,
                                           fID,
                                           &bvhTransform,
                                           0
                                         )
             )
          {
              humanPose.roll=180.0;//(float)  (rand()%90);
              humanPose.pitch=180.0;//(float) (rand()%90);
              humanPose.yaw=180.0;//(float)   (rand()%90);
              humanPose.x = bvhTransform.joint[0].localToWorldTransformation.m[3]/100;
              humanPose.y = bvhTransform.joint[0].localToWorldTransformation.m[7]/100;
              humanPose.z = 15+ bvhTransform.joint[0].localToWorldTransformation.m[11]/100;
          }

     //Do axis rendering
     doSkeletonDraw(
                     programID,
                     programFrameBufferID,
                     FramebufferName,
                     renderedTexture,
                     renderedDepth,
                     &humanPose,
                     &indexedHumanModel,
                     &axisModel,
                     WIDTH,
                     HEIGHT,
                     0
                   );
    } else
    if (axisRendering == 2)
    {
      doBVHDraw(
                 programID,
                 programFrameBufferID,
                 FramebufferName,
                 renderedTexture,
                 renderedDepth,
                 &humanPose,
                 &mc,
                 fID,
                 &axisModel,
                 WIDTH,
                 HEIGHT,
                 0
               );
    } else
    { //Do regular skinned model rendering
     doDrawing(
                programID,
                programFrameBufferID,
                FramebufferName,
                renderedTexture,
                renderedDepth,
                &humanPose,
                //-------------
                &gpuEyelashes,
                &gpuEyebrows,
                &gpuHair,
                &gpuEyes,
                &gpuHuman,
                //-------------
                WIDTH,
                HEIGHT,
                0
              );
    }


     tri_deallocModelInternals(&humanModel);
     tri_deallocModelInternals(&eyeModel);
     tri_deallocModelInternals(&hairModel);
     tri_deallocModelInternals(&eyebrowsModel);
     usleep(1);

     if  (rgb!=0)
     {
       if (downloadOpenGLColor(rgb,0,0,WIDTH,HEIGHT))
       {
           if ((dumpVideo) || (dumpSnapshot))
           {
            char filename[512]={0};
            snprintf(filename,512,"colorFrame_0_%05u.pnm",fID);
            saveRawImageToFileOGLR(filename,rgb,WIDTH,HEIGHT,3,8);
           }
       }
     }

     if (dump2DPointOutput)
     {
       getAll2DEncodedPoints(rgb,WIDTH,HEIGHT,numberOfUniqueColors);
     }

     if (flashTexturePixels)
     {
       //Retreive
       getTextureActivation(rgb,WIDTH,HEIGHT,flashX,flashY);

       if ( (flashY>=endFlashY) ) //(flashX>=endFlashX) &&
           {
             fprintf(stderr,GREEN "\n\nDone flashing pixels\n\n" NORMAL);
             break;
           }
     }

     if ( (limit>0) && (fID>=limit) )
     {
         fprintf(stderr,"Limit hit so stopping outer dataset loop\n");
         break;
     }
    } // For ever BVH frame loop

     if (limit>0)
     {
         fprintf(stderr,"Limit hit so stopping outer dataset loop\n");
         break;
     }

      if (maxFrames>1)
        { fprintf(stderr,CYAN "\n\nLooping Dataset\n\n" NORMAL); }
   }
   while ( (dumpVideo==0) && (dumpSnapshot==0) ); //If dump video is not enabled loop forever




   if(dumpSnapshot)
   {
           saveRawImageToFileOGLR(
                                   "texture.pnm",
                                   indexedHumanModel.textureData,
                                   indexedHumanModel.header.textureDataWidth,
                                   indexedHumanModel.header.textureDataHeight,
                                   indexedHumanModel.header.textureDataChannels,
                                   8
                                 );

   }

   //We free our TRI to BVH Map
   free(humanMap);

   deallocateGpuTRI(&gpuEyelashes);
   deallocateGpuTRI(&gpuEyebrows);
   deallocateGpuTRI(&gpuHair);
   deallocateGpuTRI(&gpuEyes);

   glDeleteProgram(programID);

   //If we dumped images now lets convert them to video
   if (rgb!=0)
     {
        //Also deallocate our snapshot buffer..
        free(rgb);
        int i=0;

        if (dumpVideo)
         {
           char command[512]={0};
           snprintf(command,512,"ffmpeg -framerate 30 -i colorFrame_0_%%05d.pnm -s %ux%u -y -r 30 -pix_fmt yuv420p -threads 8 lastRun3DHiRes.mp4",WIDTH,HEIGHT);
           i=system(command);
         }

        if (dumpSnapshot==0)
        {
         if(i==0)
         {
          i=system("rm *.pnm");
         }
        }
     }


   stop_glx3_stuff();
 return 0;
}

