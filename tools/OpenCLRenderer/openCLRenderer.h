/** @file openCLRenderer.h
 *  @ingroup openCLRenderer
 *  @brief This is a simple OpenCL renderer under construction
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef OPENCL_RENDERER_H_INCLUDED
#define OPENCL_RENDERER_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif


/**
 * @brief This structure holds the 3D renderer configuration, if you want to initialize this structure please use simpleRendererDefaults
 * @ingroup simpleRenderer
 */
struct oclRenderer
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


  struct Matrix4x4OfFloats projectionMatrix;
  struct Matrix4x4OfFloats viewMatrix;
  struct Matrix4x4OfFloats modelMatrix;
  struct Matrix4x4OfFloats modelViewMatrix;
  int   viewport[4];
};




/**
 * @brief This function initializes a simple renderer instance to become ready to render 3D points. The simpleRendererDefaults must be called before this call to make sure the
 *  new renderer is not initialized using wrong settings  
 * @ingroup simpleRenderer
 * @param The simpleRenderer structure that we want to initialize 
 * @retval 1=Success/0=Failure
 */
int oclr_initialize(struct oclRenderer * or);


#ifdef __cplusplus
}
#endif


#endif // SIMPLERENDERER_H_INCLUDED