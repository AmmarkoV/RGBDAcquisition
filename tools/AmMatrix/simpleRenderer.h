#ifndef SIMPLERENDERER_H_INCLUDED
#define SIMPLERENDERER_H_INCLUDED



#ifdef __cplusplus
extern "C"
{
#endif

struct simpleRenderer
{
  float fx;
  float fy;
  float skew;
  float cx;
  float cy;
  float near;
  float far;
  float width;
  float height;

  float cameraOffsetPosition[4];
  float cameraOffsetRotation[4];
  int removeObjectPosition;


  float projectionMatrix[16];
  float viewMatrix[16];
  float modelMatrix[16];
  float modelViewMatrix[16];
  int   viewport[4];
};


int simpleRendererRenderUsingPrecalculatedMatrices(
                                                    struct simpleRenderer * sr ,
                                                    float * position3D,
                                                    ///---------------
                                                    float * output2DX,
                                                    float * output2DY,
                                                    float * output2DW
                                                  );


int simpleRendererRender(
                         struct simpleRenderer * sr ,
                         float * position3D,
                         float * center3D,
                         float * objectRotation,
                         unsigned int rotationOrder, 
                         ///---------------
                         float * output2DX,
                         float * output2DY,
                         float * output2DW
                        );


int simpleRendererDefaults(
                            struct simpleRenderer * sr,
                            unsigned int width,
                            unsigned int height,
                            float fX,
                            float fY
                          );

int simpleRendererInitialize(struct simpleRenderer * sr);


int simpleRendererInitializeFromExplicitConfiguration(struct simpleRenderer * sr);



#ifdef __cplusplus
}
#endif


#endif // SIMPLERENDERER_H_INCLUDED
