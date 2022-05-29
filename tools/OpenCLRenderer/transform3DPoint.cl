/*
  resultPoint3D[0] = m[e3]  * W + m[e0]  * X + m[e1]  * Y + m[e2]  * Z;
  resultPoint3D[1] = m[e7]  * W + m[e4]  * X + m[e5]  * Y + m[e6]  * Z;
  resultPoint3D[2] = m[e11] * W + m[e8]  * X + m[e9]  * Y + m[e10] * Z;
  resultPoint3D[3] = m[e15] * W + m[e12] * X + m[e13] * Y + m[e14] * Z;
*/
__kernel void transform3DPoint(
                                __global float * transformation4x4,
                                __global float * input3DPoints,
                                __global int *   indices,
                                __global float * result3DPoints
                              )
{
  int idx = get_global_id(0);
  int idy = get_global_id(1);
  //-------------------------------------------------------------------------------
  int mIdx = idx*16; //Index of input Matrix work item
  int vIdx = idx*4;  //Index of input 3D point work item
  int vIdy = idy*4;  //Index of output 3D point work item
  //-------------------------------------------------------------------------------
  float * m = &transformation4x4[mIdx];
  //-------------------------------------------------------------------------------
  float X=input3DPoints[vIdx+0];
  float Y=input3DPoints[vIdx+1];
  float Z=input3DPoints[vIdx+2];
  float W=input3DPoints[vIdx+3];
  //-------------------------------------------------------------------------------
  int indxW = indices[vIdy+0];
  int indxX = indices[vIdy+1];
  int indxY = indices[vIdy+2];
  int indxZ = indices[vIdy+3];
  //-------------------------------------------------------------------------------
  result3DPoints[vIdx+idy] = m[indxW] * W + m[indxX] * X + m[indxY] * Y + m[indxZ] * Z;
  //-------------------------------------------------------------------------------
}




/*
__kernel void transform3DPoint(
                                __global float * transformation4x4,
                                __global float * input3DPoints,
                                __global int *   indices,
                                __global float * result3DPoints
                              )
{
  int idx = get_global_id(0);
  int idy = get_global_id(1);
  //-------------------------------------------------------------------------------
  float * m = &transformation4x4[idx*16];
  float X=input3DPoints[idx*4+0];
  float Y=input3DPoints[idx*4+1];
  float Z=input3DPoints[idx*4+2];
  float W=input3DPoints[idx*4+3];
  //-------------------------------------------------------------------------------  
  float * resultPoint3D = &result3DPoints[idx*4];
  //-------------------------------------------------------------------------------
  resultPoint3D[0] = m[3]  * W + m[0]  * X + m[1]  * Y + m[2]  * Z;
  resultPoint3D[1] = m[7]  * W + m[4]  * X + m[5]  * Y + m[6]  * Z;
  resultPoint3D[2] = m[11] * W + m[8]  * X + m[9]  * Y + m[10] * Z;
  resultPoint3D[3] = m[15] * W + m[12] * X + m[13] * Y + m[14] * Z;
  //-------------------------------------------------------------------------------

  // Ok we have our results but now to normalize our vector
  if (resultPoint3D[3]!=0.0)
  {
   resultPoint3D[0]/=resultPoint3D[3];
   resultPoint3D[1]/=resultPoint3D[3];
   resultPoint3D[2]/=resultPoint3D[3];
   resultPoint3D[3]=1.0; // resultPoint3D[3]/=resultPoint3D[3];
  }  
}
*/
