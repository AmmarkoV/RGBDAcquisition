/** @file model_loader_transform_joints.c
 *  @brief  Recursive node transformations for TRIModels part of
            https://github.com/AmmarkoV/RGBDAcquisition/tree/master/opengl_acquisition_shared_library/opengl_depth_and_color_renderer
 *  @author Ammar Qammaz (AmmarkoV)
 */
#include "model_loader_tri.h"
#include "model_loader_transform_joints.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../../../../../tools/AmMatrix/matrix4x4Tools.h"
#include "../../../../../tools/AmMatrix/quaternions.h"

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */

static float _triTrans_degrees_to_rad(float degrees)
{
    return degrees * (M_PI /180.0 );
}

static void _triTrans_create4x4MatrixFromEulerAnglesZYX(float * m ,float eulX, float eulY, float eulZ)
{
    //roll = X , pitch = Y , yaw = Z
    float x = _triTrans_degrees_to_rad(eulX);
    float y = _triTrans_degrees_to_rad(eulY);
    float z = _triTrans_degrees_to_rad(eulZ);

	float cr = cos(z);
	float sr = sin(z);
	float cp = cos(y);
	float sp = sin(y);
	float cy = cos(x);
	float sy = sin(x);

	float srsp = sr*sp;
	float crsp = cr*sp;

	m[0] = (float) cr*cp;
	m[1] = (float) crsp*sy - sr*cy;
	m[2] = (float) crsp*cy + sr*sy;
	m[3] = 0.0;  // 4x4

	m[4] = (float) sr*cp;
	m[5] = (float) srsp*sy + cr*cy;
	m[6] = (float) srsp*cy - cr*sy;
	m[7] = 0.0;  // 4x4

	m[8] = (float) -sp;
	m[9] = (float) cp*sy;
	m[10]= (float) cp*cy;
	m[11]= 0.0;  // 4x4

     // 4x4 last row
	m[12]= 0.0;
	m[13]= 0.0;
	m[14]= 0.0;
	m[15]= 1.0;
}

/// Clamp a value to 0-255
int Clamp(int i)
{
  if (i < 0) return 0;
  if (i > 255) return 255;
  return i;
}

/// h is from 0-360
/// s,v values are 0-1
/// r,g,b values are 0-255
void HsvToRgb(float h,float S,float V, float * r, float * g, float * b)
{
  // ######################################################################
  // T. Nathan Mundhenk
  // mundhenk@usc.edu
  // C/C++ Macro HSV to RGB

  float H = h;
  while (H < 0) { H += 360; };
  while (H >= 360) { H -= 360; };
  float R, G, B;
  if (V <= 0)
    { R = G = B = 0; }
  else if (S <= 0)
  {
    R = G = B = V;
  }
  else
  {
    float hf = H / 60.0;
    int i = (int) hf;
    float f = hf - i;
    float pv = V * (1 - S);
    float qv = V * (1 - S * f);
    float tv = V * (1 - S * (1 - f));
    switch (i)
    {
      // Red is the dominant color
      case 0:  R = V;  G = tv; B = pv; break;

      // Green is the dominant color
      case 1:  R = qv; G = V;  B = pv; break;
      case 2:  R = pv; G = V;  B = tv; break;

      // Blue is the dominant color
      case 3:  R = pv; G = qv; B = V; break;
      case 4:  R = tv; G = pv; B = V; break;

      // Red is the dominant color
      case 5:  R = V; G = pv; B = qv; break;

      // Just in case we overshoot on our math by a little, we put these here. Since its a switch it won't slow us down at all to put these here.
      case 6:  R = V; G = tv; B = pv; break;
      case -1: R = V; G = pv; B = qv; break;

      // The color is not defined, we should throw an error.
      default:
        //LFATAL("i Value error in Pixel conversion, Value is %d", i);
        R = G = B = V; // Just pretend its black/white
        break;
    }
  }
  *r = (float) Clamp((int)(R * 255.0));
  *g = (float) Clamp((int)(G * 255.0));
  *b = (float) Clamp((int)(B * 255.0));
}

void getDistinctColor3F_ForID(unsigned int id,unsigned maxID , float *oR,float *oG,float *oB)
{
  unsigned int sCoef=10;
  unsigned int vCoef=40;

  unsigned int hStep = (unsigned int) 360/maxID;
  unsigned int sStep = (unsigned int) sCoef/maxID;
  unsigned int vStep = (unsigned int) vCoef/maxID;

  // assumes hue [0, 360), saturation [0, 100), lightness [0, 100)
  float h = id * hStep ;
  float s = (100-sCoef) + ( (sStep-sCoef) * sStep );
  float v = (100-vCoef) + ( (vStep-vCoef) * vStep );

  HsvToRgb(h, (float) s/100, (float) v/100 , oR , oG , oB);

  *oR = (float) *oR / 255;
  *oG = (float) *oG / 255;
  *oB = (float) *oB / 255;
}

float * generatePalette(struct TRI_Model * in)
{
  unsigned int maxNumBones = in->header.numberOfBones;
  fprintf(stderr,"generating palette for model with %u bones \n",maxNumBones);
  unsigned int i=0 ;
  float * gp = (float*) malloc(sizeof(float)*maxNumBones * 3);

  if (gp!=0)
  {
   for (i=0; i<maxNumBones; i++)
   {
     getDistinctColor3F_ForID(
                              i,
                              maxNumBones,
                              &gp[3*i+0],
                              &gp[3*i+1],
                              &gp[3*i+2]
                             );
   }
  }

 return gp;
}
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
struct TRI_Bones_Per_Vertex * allocTransformTRIBonesToVertexBoneFormat(struct TRI_Model * in)
{
  struct TRI_Bones_Per_Vertex * out = (struct TRI_Bones_Per_Vertex *) malloc(sizeof(struct TRI_Bones_Per_Vertex));
  if (out==0) { return 0; }

  out->numberOfBones = in->header.numberOfBones;
  out->numberOfVertices = in->header.numberOfVertices;
  out->bonesPerVertex = (struct TRI_Bones_Per_Vertex_Vertice_Item *)   malloc(sizeof(struct TRI_Bones_Per_Vertex_Vertice_Item) *  in->header.numberOfVertices);
  if (out->bonesPerVertex==0) { free(out); return 0; }

  memset(out->bonesPerVertex,0,sizeof(struct TRI_Bones_Per_Vertex_Vertice_Item) *  in->header.numberOfVertices);

  unsigned int b=0 , w=0 , outOfSpace=0;
  for (b=0; b<in->header.numberOfBones; b++)
  {
      for (w=0; w<in->bones[b].info->boneWeightsNumber; w++)
      {
         unsigned int boneIndex = in->bones[b].weightIndex[w];
         struct TRI_Bones_Per_Vertex_Vertice_Item * bone =  &out->bonesPerVertex[boneIndex];


          if (bone->bonesOfthisVertex < MAX_BONES_PER_VERTICE)
          {
            bone->indicesOfThisVertex[bone->bonesOfthisVertex] = in->bones[b].weightIndex[w];
            bone->weightsOfThisVertex[bone->bonesOfthisVertex] = in->bones[b].weightValue[w];
            bone->boneIDOfThisVertex[bone->bonesOfthisVertex]  = b;

            ++bone->bonesOfthisVertex;
          } else
          {
            ++outOfSpace;
          }
      }
  }

  if (outOfSpace>0)
  {
    fprintf(stderr,"Vertices are set up to accomodate at most %d bones , %u vertices where too small for our input .. \n",MAX_BONES_PER_VERTICE,outOfSpace);
  }
 return out;
}

void freeTransformTRIBonesToVertexBoneFormat(struct TRI_Bones_Per_Vertex * in)
 {
   if (in!=0)
   {
     if (in->bonesPerVertex!=0)
     {
      free(in->bonesPerVertex);
     }
    free(in);
   }
 }
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
unsigned int * convertTRIBonesToParentList(struct TRI_Model * in , unsigned int * outputNumberOfBones)
{

  *outputNumberOfBones =  in->header.numberOfBones;
  unsigned int * parentNode = ( unsigned int * )  malloc (sizeof(unsigned int) * in->header.numberOfBones);

if (parentNode!=0)
{
   memset(parentNode,0,sizeof(unsigned int) * in->header.numberOfBones);

   unsigned int i=0;
   for (i=0; i< in->header.numberOfBones; i++)
   {
     parentNode[i] = in->bones[i].info->boneParent;
   }
}
 return parentNode;
}

//Please note that the output is in the coordinate space of the binding pose model and needs to be transformed/projected etc
//according to the real location of the mesh
float * convertTRIBonesToJointPositions(struct TRI_Model * in , unsigned int * outputNumberOfJoints)
{
  if (in==0)                   { return 0; }
  if (outputNumberOfJoints==0) { return 0; }

  float * outputJoints = ( float * )  malloc (sizeof(float) * in->header.numberOfBones * 3);
  if(outputJoints!=0)
  {
  memset(outputJoints,0,sizeof(float) * in->header.numberOfBones * 3);

  unsigned int * outputNumberSamples = ( unsigned int * )  malloc (sizeof(unsigned int ) * in->header.numberOfBones);
  if (outputNumberSamples==0)
  {
    fprintf(stderr,"Could not allocate internal sample space , convertTRIBonesToJointPositions will fail \n");
    free(outputJoints);
    *outputNumberOfJoints=0;
    return 0;
  }
  memset(outputNumberSamples,0,sizeof(unsigned int) * in->header.numberOfBones);

  *outputNumberOfJoints =  in->header.numberOfBones;

  struct TRI_Bones_Per_Vertex * bpv = allocTransformTRIBonesToVertexBoneFormat(in);
  if (bpv!=0)
  {
   unsigned int i=0;
   for (i=0; i<in->header.numberOfVertices; i++)
    {
      struct TRI_Bones_Per_Vertex_Vertice_Item * bone =  &bpv->bonesPerVertex[i];

      float maxWeight = 0.0;
      unsigned int z=0 , b=0;
      for (z=0; z<bone->bonesOfthisVertex; z++)
      {
        if ( bone->weightsOfThisVertex[z]>maxWeight )
            {
              maxWeight=bone->weightsOfThisVertex[z];
              b=z;
            }
      }

      unsigned int indxID=bone->indicesOfThisVertex[b];
      unsigned int boneID=bone->boneIDOfThisVertex[b];

      if (boneID>in->header.numberOfVertices)
      {
         fprintf(stderr,"Error bug detected \n"); boneID=0;
      }

      ++outputNumberSamples[boneID];
      outputJoints[boneID*3+0]+=in->vertices[indxID*3+0];
      outputJoints[boneID*3+1]+=in->vertices[indxID*3+1];
      outputJoints[boneID*3+2]+=in->vertices[indxID*3+2];
   }


   for (i=0; i< in->header.numberOfBones; i++)
   {
     if (outputNumberSamples[i]>0)
     {
      outputJoints[i*3+0]=outputJoints[i*3+0]/outputNumberSamples[i];
      outputJoints[i*3+1]=outputJoints[i*3+1]/outputNumberSamples[i];
      outputJoints[i*3+2]=outputJoints[i*3+2]/outputNumberSamples[i];

      //fprintf(stderr,"Bone %u (%s) = ",i,in->bones[i].boneName);
      //fprintf(stderr," %0.2f,%0.2f,%0.2f  ( %u samples ) \n ",outputJoints[i*3+0],outputJoints[i*3+1],outputJoints[i*3+2],outputNumberSamples[i]);
     } else
     {
      //fprintf(stderr,"Bone %u (%s) has no samples.. \n",i,in->bones[i].boneName);
     }
   }


   freeTransformTRIBonesToVertexBoneFormat(bpv);
  }

 free(outputNumberSamples);

 }
 return outputJoints;
}

unsigned int  * getClosestVertexToJointPosition(struct TRI_Model * in , float * joints , unsigned int numberOfJoints)
{
  fprintf(stderr,"getClosestVertexToJointPosition \n");
  unsigned int * outputPositions = ( unsigned int * )  malloc (sizeof(unsigned int ) * numberOfJoints );

  if (outputPositions!=0)
  {
   memset(outputPositions,0,sizeof(unsigned int ) * numberOfJoints);

   unsigned int i=0,v=0;
   float x,y,z,diffX,diffY,diffZ , bestDistance=10000000000 , worstDistance=0 , currentDistance;

   fprintf(stderr,"searching best vertices for %u joints \n",numberOfJoints);
   for (i=0; i<numberOfJoints; i++)
   {
     x=joints[i*3+0];
     y=joints[i*3+1];
     z=joints[i*3+2];

     if ( (x!=x) || (y!=y) || (z!=z) )
     {
       //IGNORE NAN VALUE
     } else
     {
     bestDistance=10000000000;
     worstDistance=0;
     fprintf(stderr,"now doing vertice search among %u vertices \n",in->header.numberOfVertices/3);
     for (v=0; v<in->header.numberOfVertices/3; v++)
     {
       diffX=x-in->vertices[v*3+0];
       diffY=y-in->vertices[v*3+1];
       diffZ=z-in->vertices[v*3+2];

       currentDistance = (diffX*diffX) + (diffY*diffY) + (diffZ*diffZ);
       if ( currentDistance < bestDistance )
       {
         outputPositions[i]=v;
         bestDistance=currentDistance;
       }
       if ( currentDistance > worstDistance )
       {
         worstDistance=currentDistance ;
       }

       if (bestDistance==0)  //Cant find a better match..
          {break;}
     }

     fprintf(stderr,"Bone %u (%s) = ",i,in->bones[i].boneName);
     fprintf(stderr," %0.2f,%0.2f,%0.2f \n ",joints[i*3+0],joints[i*3+1],joints[i*3+2]);
     fprintf(stderr," Best Vertice is %u with a distance of %0.2f ( worst %0.2f ) \n ",outputPositions[i],sqrt(bestDistance),sqrt(worstDistance));
     }
   }


  }
 return outputPositions;
}

int setTRIModelBoneInitialPosition(struct TRI_Model * in)
{
   unsigned int outputNumberOfJoints;
   float * pos = convertTRIBonesToJointPositions( in , &outputNumberOfJoints);
   if (pos!=0)
   {
    unsigned int i=0;
    for (i=0; i<in->header.numberOfBones; i++)
    {
     in->bones[i].info->x = pos[i*3+0];
     in->bones[i].info->y = pos[i*3+1];
     in->bones[i].info->z = pos[i*3+2];

     fprintf(stderr,"Bone %u (%s) = ",i,in->bones[i].boneName);
     fprintf(stderr," %0.2f,%0.2f,%0.2f \n ",pos[i*3+0],pos[i*3+1],pos[i*3+2]);
    }

    free(pos);
    return 1;
   }

   return 0;
}
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
void colorCodeBones(struct TRI_Model * in)
{
   if (in->bones==0)
   {
     fprintf(stderr,"No bones to colorcode \n");
     return;
   }

  struct TRI_Bones_Per_Vertex * bpv = allocTransformTRIBonesToVertexBoneFormat(in);
  if (bpv!=0)
  {
   float * gp = generatePalette(in);
   if (gp!=0)
   {
    if (in->colors==0)
    {
      in->header.numberOfColors = in->header.numberOfVertices;
      in->colors = (float*) malloc(sizeof(float) * in->header.numberOfColors );
    }

   if  (
        (in->colors!=0) &&
        (in->header.numberOfColors!=0) &&
        (in->header.numberOfVertices!=0)
       )
   {
   unsigned int i=0;
   for (i=0; i<in->header.numberOfVertices; i++)
    {
      struct TRI_Bones_Per_Vertex_Vertice_Item * bone =  &bpv->bonesPerVertex[i];

      float maxWeight = 0.0;
      unsigned int z=0 , b=0;
      for (z=0; z<bone->bonesOfthisVertex; z++)
      {
        if ( bone->weightsOfThisVertex[z]>maxWeight )
            {
              maxWeight=bone->weightsOfThisVertex[z];
              b=z;
            }
      }

      unsigned int indxID=bone->indicesOfThisVertex[b];
      unsigned int boneID=bone->boneIDOfThisVertex[b];
      if (boneID>in->header.numberOfVertices)
      {
         fprintf(stderr,"Error bug detected \n"); boneID=0;
      }

      in->colors[indxID*3+0]=gp[boneID*3+0];
      in->colors[indxID*3+1]=gp[boneID*3+1];
      in->colors[indxID*3+2]=gp[boneID*3+2];
    }
   }
   free(gp);
   }
   freeTransformTRIBonesToVertexBoneFormat(bpv);
  }
}


/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
int tri_colorCodeTexture(struct TRI_Model * in, unsigned int x, unsigned int y, unsigned int width,unsigned int height)
{
  if ( (width==0) || (height==0) ) { return 0; }

  unsigned int colorStep = (255*255*255) / (width * height);
  unsigned int currentColor = colorStep;
  //------------------------------------
  if (in!=0)
  {
    if (in->textureData!=0)
    {
      unsigned int imageWidth  = in->header.textureDataWidth;
      //unsigned int imageHeight = in->header.textureDataHeight;

      unsigned int x1 = x;
      unsigned int y1 = y;
      unsigned int x2 = x + width;
      unsigned int y2 = y + height;

      char * ptr = in->textureData + (y1 * imageWidth * 3) + (x1 * 3);
      char * lineStart = ptr;
      char * lineEnd   = ptr + (width * 3);
      char * imageEnd  = in->textureData + (y2 * imageWidth * 3) + (x2 * 3);

      while (ptr<imageEnd)
      {
       while (ptr<lineEnd)
       {
        unsigned int thisPixel = currentColor;
        //------------------------------------
        char c1 = (char) thisPixel % 255;
        thisPixel = thisPixel / 255;
        char c2 = (char) thisPixel % 255;
        thisPixel = thisPixel / 255;
        char c3 = (char)  thisPixel % 255;
        //------------------------------------
        *ptr = c3;  ++ptr; // R
        *ptr = c2;  ++ptr; // G
        *ptr = c1;  ++ptr; // B
        //---------------------------
        currentColor += colorStep;
       }
       //--------------------------
       lineStart += imageWidth * 3;
       lineEnd   += imageWidth * 3;
       ptr = lineStart;
       //--------------------------
      }

      return 1;
    }
  }
  return 0;
}
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------


int setTRIJointRotationOrder(
                             struct TRI_Model * in ,
                             unsigned int jointToChange ,
                             unsigned int rotationOrder
                           )
{
  if (in==0)        { return 0; }
  if (in->bones==0) { return 0; }

  if ( jointToChange < in->header.numberOfBones )
  {
    fprintf(stderr,"setTRIJointRotationOrder : joint=%u -> order=%u ",jointToChange,rotationOrder);
    if (rotationOrder<ROTATION_ORDER_NUMBER_OF_NAMES)
    {
       fprintf(stderr," %s \n",ROTATION_ORDER_NAMESA[rotationOrder]);
    } else
    {
       fprintf(stderr,"\n");
    }

    in->bones[jointToChange].info->eulerRotationOrder = (unsigned char) rotationOrder;
    return 1;
  }
 return 0;
}

int getTRIJointRotationOrder(
                              struct TRI_Model * in ,
                              unsigned int jointToChange ,
                              unsigned int rotationOrder
                            )
{
  if (in==0)        { return 0; }
  if (in->bones==0) { return 0; }

  unsigned char val =  in->bones[jointToChange].info->eulerRotationOrder;

  return (int) val;
}

void transformTRIJoint(
                        struct TRI_Model * in ,
                        float * jointData ,
                        unsigned int jointDataSize ,

                        unsigned int jointToChange ,
                        float rotEulerX , //Roll
                        float rotEulerY , //Pitch
                        float rotEulerZ   //Yaw
                      )
{
  //This is needed for meta reasons
  in->bones[jointToChange].info->rotX = rotEulerX;
  in->bones[jointToChange].info->rotY = rotEulerY;
  in->bones[jointToChange].info->rotZ = rotEulerZ;

  //We set the 4x4 Matrix that is what is used for the transform..
  float * mat = &jointData[16*jointToChange];

  _triTrans_create4x4MatrixFromEulerAnglesZYX(mat,rotEulerX,rotEulerY,rotEulerZ);
}

float * mallocModelTransformJoints(
                                    struct TRI_Model * triModelInput ,
                                    unsigned int * jointDataSizeOutput
                                   )
{
  float * returnMat = (float * ) malloc(sizeof(float) * 16 * triModelInput->header.numberOfBones);
  if (returnMat)
  {
     *jointDataSizeOutput =  triModelInput->header.numberOfBones;
     unsigned int i=0;
     for (i=0; i<(*jointDataSizeOutput); i++)
     {
       float * m = &returnMat[16*i];
       create4x4FIdentityMatrixDirect(m);
       //m[0] = 1.0;  m[1] = 0.0;  m[2] = 0.0;   m[3] = 0.0;
       //m[4] = 0.0;  m[5] = 1.0;  m[6] = 0.0;   m[7] = 0.0;
       //m[8] = 0.0;  m[9] = 0.0;  m[10] = 1.0;  m[11] =0.0;
       //m[12]= 0.0;  m[13]= 0.0;  m[14] = 0.0;  m[15] = 1.0;
     }
  }
  return returnMat;
}

float * mallocModelTransformJointsEulerAnglesDegrees(
                                                      struct TRI_Model * triModelInput ,
                                                      float * jointData ,
                                                      unsigned int jointDataSize ,
                                                      unsigned int method
                                                     )
{
 float * returnMat = (float * ) malloc(sizeof(float) * 3 * triModelInput->header.numberOfBones);
 if (returnMat)
  {
     float euler[4]={0};
     float quaternions[4]={0};
     float m4x4[16]={0};

     unsigned int i=0;
     for (i=0; i<jointDataSize; i++)
     {
       float * mat = &jointData[16*i];

       copy4x4FMatrix(m4x4,mat);
       matrix4x42Quaternion(quaternions,qXqYqZqW,m4x4);

       quaternions2Euler(euler,quaternions,qXqYqZqW);

       returnMat[i*3+0] = euler[0];
       returnMat[i*3+1] = euler[1];
       returnMat[i*3+2] = euler[2];
     }
  }
  return returnMat;
}

void printModelTransform(struct TRI_Model * in)
{
  for (unsigned int i=0; i<in->header.numberOfBones; i++)
    {
      if (in->bones[i].info->altered)
      {
        fprintf(stderr,"POSE4x4(this,0,%s",in->bones[i].boneName);

        for (unsigned int z=0; z<16; z++)
        {
          fprintf(stderr,",%0.3f",in->bones[i].info->finalVertexTransformation[z]);
        }
        fprintf(stderr,")\n");
      }
    }
}

/* This is direct setting of the joint data , overwriting default values */
void recursiveJointHierarchyTransformer(
                                        struct TRI_Model * in  ,
                                        int curBone ,
                                        struct Matrix4x4OfFloats parentTransformUntouched,
                                        float * joint4x4Data , unsigned int joint4x4DataSize ,
                                        unsigned int recursionLevel
                                       )
{
  //Sanity check..
  //-----------------------------
  if (in==0) { return; }
  //-----------------------------
  if (recursionLevel>=in->header.numberOfBones+1)
        { fprintf(stderr,RED "BUG : REACHED RECURSION LIMIT (%u/%u)\n" NORMAL,recursionLevel,in->header.numberOfBones); return; }
  //-----------------------------

  struct Matrix4x4OfFloats globalTransformation={0};
  struct Matrix4x4OfFloats nodeTransformation  ={0};
  struct Matrix4x4OfFloats parentTransform     ={0};
  copy4x4FMatrixToAlignedContainer(&parentTransform,parentTransformUntouched.m);


  //We use nodeLocalTransformation as shorthand so that we don't have to access the bone structure every time
  struct Matrix4x4OfFloats nodeLocalTransformation;
  copy4x4FMatrixToAlignedContainer(&nodeLocalTransformation,in->bones[curBone].info->localTransformation);

  if ( (joint4x4Data!=0) && (curBone*16<joint4x4DataSize) )
     {
       struct Matrix4x4OfFloats joint4x4DataPacked   ={0};
       copy4x4FMatrixToAlignedContainer(&joint4x4DataPacked,&joint4x4Data[curBone*16]);
       //We do the transformation of our node with the new joint 4x4 data we received..!  nodeTransformationCopy
       multiplyTwo4x4FMatricesS(&nodeTransformation,&nodeLocalTransformation,&joint4x4DataPacked);

     } else
     {
       //If there is no 4x4 transform to use then just copy our local transformation
       fprintf(stderr,YELLOW "Bone %u has no joint transform.. \n" NORMAL,curBone);
       copy4x4FMatrix(nodeTransformation.m,nodeLocalTransformation.m);
     }

  //We calculate the globalTransformation of the node by chaining it to its parent..!
  multiplyTwo4x4FMatricesS(
                           &globalTransformation,
                           &parentTransform,
                           &nodeTransformation
                          );

  //Since we have everything ready, let's store the bone position..
  struct Vector4x1OfFloats boneCenter={0}; boneCenter.m[3]=1.0;
  transform3DPointFVectorUsing4x4FMatrix(&boneCenter,&globalTransformation,&boneCenter);
  in->bones[curBone].info->x = boneCenter.m[0];
  in->bones[curBone].info->y = boneCenter.m[1];
  in->bones[curBone].info->z = boneCenter.m[2];


  //We calculate the finalVertexTransformation for all vertices that are influenced for this bone
  //by chaining the global transformation with the bone's global inverse transform and and rest/bind pose transform
  //Apply the rotation matrix on top of the default one (inverse rot of the matrixThatTransformsFromMeshSpaceToBoneSpaceInBindPose)
  struct Matrix4x4OfFloats finalVertexTransformation;
  struct Matrix4x4OfFloats boneGlobalInverseTransform;
  struct Matrix4x4OfFloats matrixThatTransformsFromMeshSpaceToBoneSpaceInBindPose;

  copy4x4FMatrixToAlignedContainer(&matrixThatTransformsFromMeshSpaceToBoneSpaceInBindPose,&in->bones[curBone].info->matrixThatTransformsFromMeshSpaceToBoneSpaceInBindPose);
  copy4x4FMatrixToAlignedContainer(&boneGlobalInverseTransform,in->header.boneGlobalInverseTransform); //This is repeated many times for no reason

  multiplyThree4x4FMatrices(
                            &finalVertexTransformation ,
                            &boneGlobalInverseTransform ,
                            &globalTransformation,
                            &matrixThatTransformsFromMeshSpaceToBoneSpaceInBindPose
                           );

  //Copy back our calculated output to the TRI bones
  copy4x4FMatrix(in->bones[curBone].info->finalVertexTransformation,&finalVertexTransformation);


   //Each bone might have multiple children, we recursively execute the same transform for all children of this node..!
  for (unsigned int childID = 0; childID < in->bones[curBone].info->numberOfBoneChildren; childID++)
      {
        recursiveJointHierarchyTransformer(
                                            in,
                                            in->bones[curBone].boneChild[childID], //recursively execute on this bone's child
                                            globalTransformation,
                                            joint4x4Data,joint4x4DataSize,
                                            recursionLevel+1
                                           );
      }

  return;
}









/* This is direct setting of the joint data , overwriting default values */
void recursiveJointHierarchyTransformerDirect(
                                               struct TRI_Model * in  ,
                                               int curBone ,
                                               float * parentTransformUntouched ,
                                               float * joint4x4Data , unsigned int joint4x4DataSize ,
                                               unsigned int recursionLevel
                                             )
{
  //Sanity check..
  //-----------------------------
  if (in==0) { return; }
  //-----------------------------
  if (recursionLevel>=in->header.numberOfBones+1)
        { fprintf(stderr,RED "BUG : REACHED RECURSION LIMIT (%u/%u)\n" NORMAL,recursionLevel,in->header.numberOfBones); return; }
  //-----------------------------

  float emptyParentTransform[16], globalTransformation[16], nodeTransformation[16];
  float * parentTransform = parentTransformUntouched;

  //int multiplyThree4x4FMatrices(struct Matrix4x4OfFloats * result,struct Matrix4x4OfFloats * matrixA,struct Matrix4x4OfFloats * matrixB,struct Matrix4x4OfFloats * matrixC);

  if (parentTransformUntouched==0)
   {
      //If parentTransformUntouched is empty then use an identity matrix locally allocated in our emptyParentTransform
      //as a parentTransform, in any case we do not touch parentTransformUntouched..!
      create4x4FIdentityMatrixDirect((float*) &emptyParentTransform);
      parentTransform = emptyParentTransform;
   }

  //We use nodeLocalTransformation as shorthand so that we don't have to access the bone structure every time
  float * nodeLocalTransformation = in->bones[curBone].info->localTransformation;

  if ( (joint4x4Data!=0) && (curBone*16<joint4x4DataSize) )
     {
       //We do the transformation of our node with the new joint 4x4 data we received..!  nodeTransformationCopy
       multiplyTwo4x4FMatrices_Naive(nodeTransformation,nodeLocalTransformation,&joint4x4Data[curBone*16]);
     } else
     {
       //If there is no 4x4 transform to use then just copy our local transformation
       fprintf(stderr,YELLOW "Bone %u has no joint transform.. \n" NORMAL,curBone);
       copy4x4FMatrix(nodeTransformation,nodeLocalTransformation);
     }

  //We calculate the globalTransformation of the node by chaining it to its parent..!
  multiplyTwo4x4FMatrices_Naive(
                                 globalTransformation,
                                 parentTransform,
                                 nodeTransformation
                               );

  //Since we have everything ready, let's store the bone position..
  struct Vector4x1OfFloats boneCenter={0}; boneCenter.m[3]=1.0;
  transform3DPointFVectorUsing4x4FMatrix_Naive(boneCenter.m,globalTransformation,boneCenter.m);
  in->bones[curBone].info->x = boneCenter.m[0];
  in->bones[curBone].info->y = boneCenter.m[1];
  in->bones[curBone].info->z = boneCenter.m[2];


  //We calculate the finalVertexTransformation for all vertices that are influenced for this bone
  //by chaining the global transformation with the bone's global inverse transform and and rest/bind pose transform
  //Apply the rotation matrix on top of the default one (inverse rot of the matrixThatTransformsFromMeshSpaceToBoneSpaceInBindPose)
  multiplyThree4x4FMatrices_Naive(
                                   in->bones[curBone].info->finalVertexTransformation ,
                                   in->header.boneGlobalInverseTransform ,
                                   globalTransformation,
                                   in->bones[curBone].info->matrixThatTransformsFromMeshSpaceToBoneSpaceInBindPose
                                  );

   //Each bone might have multiple children, we recursively execute the same transform for all children of this node..!
  for (unsigned int childID = 0; childID < in->bones[curBone].info->numberOfBoneChildren; childID++)
      {
        recursiveJointHierarchyTransformerDirect(
                                                 in,
                                                 in->bones[curBone].boneChild[childID], //recursively execute on this bone's child
                                                 globalTransformation,
                                                 joint4x4Data,joint4x4DataSize,
                                                 recursionLevel+1
                                                );
      }

  return;
}


int applyVertexTransformation( struct TRI_Model * triModelOut , struct TRI_Model * triModelIn )
{
   if ( (triModelIn->vertices==0) && (triModelIn->normal==0) )
       {
         fprintf(stderr,YELLOW "Cannot apply vertex transformation with no vertices or normals.. \n" NORMAL);
         return 0;
       }

  struct Vector4x1OfFloats transformedPosition={0},transformedNormal={0},position={0},normal={0};

  //We NEED to clear the vertices and normals since they are added uppon , not having
  //the next two lines results in really weird and undebuggable visual behaviour
  memset(triModelOut->vertices,0,triModelOut->header.numberOfVertices * sizeof(float));

  //Clean normal output before repopulating it..
  if (triModelIn->normal!=0)
       {
         memset(triModelOut->normal,0,triModelOut->header.numberOfNormals * sizeof(float));
       }

  //We will need a Matrix4x4OfFloats since it is aligned to give SSE speedups..
  struct  Matrix4x4OfFloats boneTransformMatrix;

  for (unsigned int boneID=0; boneID<triModelIn->header.numberOfBones; boneID++)
   {
     if ( is4x4FZeroMatrix(triModelIn->bones[boneID].info->finalVertexTransformation) )
     {
       fprintf(stderr,RED "Joint Transform was zero for bone %s (%u) , there was a bug preparing the matrices \n" NORMAL,triModelIn->bones[boneID].boneName , boneID );
       float * m = triModelIn->bones[boneID].info->finalVertexTransformation;
       create4x4FIdentityMatrixDirect(m);
     }

     //CPU bone transformations..!
     for (unsigned int boneWeightID=0; boneWeightID<triModelIn->bones[boneID].info->boneWeightsNumber; boneWeightID++)
     {
       //V is the vertice we will be working in this loop
       unsigned int vertexID = triModelIn->bones[boneID].weightIndex[boneWeightID];
       //W is the weight that we have for the specific bone
       float boneWeightValue = triModelIn->bones[boneID].weightValue[boneWeightID];

       //Vertice transformation ----------------------------------------------
       //We load our input into position/normal
       position.m[0] = triModelIn->vertices[vertexID*3+0];
       position.m[1] = triModelIn->vertices[vertexID*3+1];
       position.m[2] = triModelIn->vertices[vertexID*3+2];
       position.m[3] = 1.0;

       //Keep a copy of our matrix on our SSE aligned Matrix4x4OfFloats structure
       copy4x4FMatrix(boneTransformMatrix.m,triModelIn->bones[boneID].info->finalVertexTransformation);

       //We transform input (initial) position with the transform we computed to get transformedPosition
       transform3DPointFVectorUsing4x4FMatrix(&transformedPosition,&boneTransformMatrix,&position);
       //Please note that triModelOut->vertices is set to 0 by memset call above so it is clean..!
       triModelOut->vertices[vertexID*3+0] += (float) transformedPosition.m[0] * boneWeightValue;
       triModelOut->vertices[vertexID*3+1] += (float) transformedPosition.m[1] * boneWeightValue;
       triModelOut->vertices[vertexID*3+2] += (float) transformedPosition.m[2] * boneWeightValue;
       //----------------------------------------------------------------------

       //Normal transformation ----------------------------------------------
       if (triModelIn->normal!=0)
       {
        normal.m[0]   = triModelIn->normal[vertexID*3+0];
        normal.m[1]   = triModelIn->normal[vertexID*3+1];
        normal.m[2]   = triModelIn->normal[vertexID*3+2];
        normal.m[3]   = 0.0;

        //We transform input (initial) normal with the transform we computed to get transformedNormal
        transform3DNormalVectorUsing3x3FPartOf4x4FMatrix(transformedNormal.m,&boneTransformMatrix,normal.m);
        triModelOut->normal[vertexID*3+0] += (float) transformedNormal.m[0] * boneWeightValue;
        triModelOut->normal[vertexID*3+1] += (float) transformedNormal.m[1] * boneWeightValue;
        triModelOut->normal[vertexID*3+2] += (float) transformedNormal.m[2] * boneWeightValue;
       }
       //----------------------------------------------------------------------
     }
   }
 return 1;
}








int doModelTransform(
                      struct TRI_Model * triModelOut,
                      struct TRI_Model * triModelIn,
                      float * joint4x4Data,
                      unsigned int joint4x4DataSize ,
                      unsigned int autodetectAlteredMatrices,//This is no longer used
                      unsigned int directSettingOfMatrices, //This is no longer used
                      unsigned int performVertexTransform, //If you want to handle the transform on a shader set this to 0
                      unsigned int jointAxisConvention
                    )
{
 if (triModelIn==0)
                     { fprintf(stderr,"doModelTransform called without input TRI Model \n"); return 0; }

 if ( ( triModelIn->vertices ==0 ) || ( triModelIn->header.numberOfVertices ==0 ) )
                     { fprintf(stderr,RED "Number of vertices is zero so can't do model transform using weights..\n" NORMAL); return 0; }

 if ( (joint4x4Data==0) || (joint4x4DataSize==0) )
 {
   fprintf(stderr,"doModelTransform called without joints to transform , ");
   fprintf(stderr,"so it will be just returning a null transformed copy of");
   fprintf(stderr,"the input mesh , hope this is what you intended..\n");
   return 0;
 }

 for (unsigned int boneID=0; boneID<triModelIn->header.numberOfBones; boneID++)
   {
     //All matrices considered and marked altered when calling this call
     triModelIn->bones[boneID].info->altered=1;
   }

  //Use simple transformer that does not rely on struct Matrix4x4OfFloats
  #define USE_SIMPLE_VERSION 0

  #if USE_SIMPLE_VERSION
  float initialParentTransform[16]={0};
  //The initial parent transform is an identity matrix..!
  create4x4FIdentityMatrixDirect((float*) &initialParentTransform);

  //This recursively calculates all matrix transforms and prepares the correct matrices in triModelIn
  //each boneID gets its final 4x4 matrix in triModelIn->bones[boneID].info->finalVertexTransformation
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  recursiveJointHierarchyTransformerDirect(
                                           triModelIn ,
                                           triModelIn->header.rootBone  ,
                                           initialParentTransform ,
                                           joint4x4Data ,
                                           joint4x4DataSize ,
                                           0 /*First call 0 level recursion*/
                                          );
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  #else
  struct Matrix4x4OfFloats parentMatrix;
  create4x4FIdentityMatrix(&parentMatrix);

  //This recursively calculates all matrix transforms and prepares the correct matrices in triModelIn
  //each boneID gets its final 4x4 matrix in triModelIn->bones[boneID].info->finalVertexTransformation
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  recursiveJointHierarchyTransformer(
                                     triModelIn ,
                                     triModelIn->header.rootBone  ,
                                     parentMatrix ,
                                     joint4x4Data ,
                                     joint4x4DataSize ,
                                     0 /*First call 0 level recursion*/
                                    );
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  #endif








  if (performVertexTransform)
  {
    //Past checks..
   tri_copyModel(triModelOut,triModelIn,1,0); //Last 1 means we also want bone data , Last 0 means we dont need to copy texture data
   applyVertexTransformation(triModelOut,triModelIn);
  }

 return 1;
}
