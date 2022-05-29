/*
  resultPoint3D[0] = m[e3]  * W + m[e0]  * X + m[e1]  * Y + m[e2]  * Z;
  resultPoint3D[1] = m[e7]  * W + m[e4]  * X + m[e5]  * Y + m[e6]  * Z;
  resultPoint3D[2] = m[e11] * W + m[e8]  * X + m[e9]  * Y + m[e10] * Z;
  resultPoint3D[3] = m[e15] * W + m[e12] * X + m[e13] * Y + m[e14] * Z;
*/

__kernel void glProjectf_ocl(__global float * result3DPoints,__global float *transformation4x4,__global float *input3DPoints,__global int * indices)
{
  float * m = transformation4x4;
  register float X=input3DPoints[0],Y=input3DPoints[1],Z=input3DPoints[2],W=input3DPoints[3];
  
  int idx = get_global_id(0);
  int idy = get_global_id(1);

  resultPoint3D[0] = m[e3]  * W + m[e0]  * X + m[e1]  * Y + m[e2]  * Z;
  resultPoint3D[1] = m[e7]  * W + m[e4]  * X + m[e5]  * Y + m[e6]  * Z;
  resultPoint3D[2] = m[e11] * W + m[e8]  * X + m[e9]  * Y + m[e10] * Z;
  resultPoint3D[3] = m[e15] * W + m[e12] * X + m[e13] * Y + m[e14] * Z;

  // Ok we have our results but now to normalize our vector
  if (resultPoint3D[3]!=0.0)
  {
   resultPoint3D[0]/=resultPoint3D[3];
   resultPoint3D[1]/=resultPoint3D[3];
   resultPoint3D[2]/=resultPoint3D[3];
   resultPoint3D[3]=1.0; // resultPoint3D[3]/=resultPoint3D[3];
   return 1;
  } else
}
